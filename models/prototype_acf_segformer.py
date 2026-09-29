import torch
import torch.nn as nn
import torch.nn.functional as F

from .mit_encoder import MiTEncoder
from .orthogonal_prototype_fusion import OrthogonalPrototypeFusion


class PrototypeACFSegFormer(nn.Module):
    def __init__(
        self,
        backbone="b3",
        pretrained_backbone=True,
        backbone_weights_path=None,
        decoder_dim=256,
        temperature=10.0,
        freeze_encoder=False,
        freeze_encoder_stages=0,
        ssp_mode="soft",
        ssp_top_ratio=0.15,
        use_ssp=True,
        ssp_blend=0.5,
        alpha_min=0.05,
        alpha_max=0.95,
        sep_margin=0.1,
        decoder_dropout=0.2,
        use_aux=True,
        use_boundary=True,
    ):
        super().__init__()
        self.encoder = MiTEncoder(
            backbone=backbone,
            pretrained_backbone=pretrained_backbone,
            backbone_weights_path=backbone_weights_path,
            freeze_encoder=freeze_encoder,
            freeze_stages=freeze_encoder_stages,
        )
        self.stage_channels = self.encoder.stage_channels
        self.prototype_fusion = OrthogonalPrototypeFusion(
            self.stage_channels,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            sep_margin=sep_margin,
        )
        self.temperature = float(temperature)
        self.use_ssp = bool(use_ssp)
        self.ssp_blend = float(ssp_blend)
        self.topk_fallback = 12

    @staticmethod
    def _resize_mask(mask, size):
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.shape[-2:] != size:
            mask = F.interpolate(mask.float(), size=size, mode="nearest")
        return (mask.float() > 0.5).float()

    @staticmethod
    def _masked_average_pooling(feature, mask):
        mask = PrototypeACFSegFormer._resize_mask(mask, feature.shape[-2:])
        denom = mask.sum(dim=(2, 3)).clamp_min(1e-6)
        proto = (feature * mask).sum(dim=(2, 3)) / denom
        return torch.nan_to_num(proto)

    def _compute_support_prototypes(self, support_features, support_masks):
        batch_size, k_shot = support_masks.shape[:2]
        masks_flat = support_masks.reshape(batch_size * k_shot, 1, *support_masks.shape[-2:])
        shot_valid = (masks_flat.flatten(1).sum(dim=1) > 0).reshape(batch_size, k_shot, 1).float()

        fg_prototypes, bg_prototypes = [], []
        fg_shot_prototypes, bg_shot_prototypes = [], []
        for feat in support_features:
            channels = feat.shape[2]
            feat_flat = feat.reshape(batch_size * k_shot, channels, *feat.shape[-2:])
            mask_stage = self._resize_mask(masks_flat, feat_flat.shape[-2:])
            fg = self._masked_average_pooling(feat_flat, mask_stage).reshape(batch_size, k_shot, channels)
            bg = self._masked_average_pooling(feat_flat, 1.0 - mask_stage).reshape(batch_size, k_shot, channels)

            denom = shot_valid.sum(dim=1).clamp_min(1.0)
            fg_proto = (fg * shot_valid).sum(dim=1) / denom
            bg_proto = bg.mean(dim=1)
            fg_prototypes.append(torch.nan_to_num(fg_proto))
            bg_prototypes.append(torch.nan_to_num(bg_proto))
            fg_shot_prototypes.append(torch.nan_to_num(fg))
            bg_shot_prototypes.append(torch.nan_to_num(bg))
        return fg_prototypes, bg_prototypes, fg_shot_prototypes, bg_shot_prototypes, shot_valid

    def _compute_mask_prototypes(self, features, masks):
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)
        fg_prototypes, bg_prototypes = [], []
        for feat in features:
            mask_stage = self._resize_mask(masks, feat.shape[-2:])
            fg_prototypes.append(self._masked_average_pooling(feat, mask_stage))
            bg_prototypes.append(self._masked_average_pooling(feat, 1.0 - mask_stage))
        return fg_prototypes, bg_prototypes

    def _cosine_logits(self, feature, fg_proto, bg_proto):
        feature = F.normalize(feature, p=2, dim=1, eps=1e-6)
        fg_proto = F.normalize(fg_proto, p=2, dim=1, eps=1e-6)[:, :, None, None]
        bg_proto = F.normalize(bg_proto, p=2, dim=1, eps=1e-6)[:, :, None, None]
        fg = (feature * fg_proto).sum(dim=1, keepdim=True)
        bg = (feature * bg_proto).sum(dim=1, keepdim=True)
        return torch.cat([bg, fg], dim=1) * self.temperature

    def _stage_logits(self, features, fg_prototypes, bg_prototypes):
        return [self._cosine_logits(feat, fg, bg) for feat, fg, bg in zip(features, fg_prototypes, bg_prototypes)]

    @staticmethod
    def _fuse_logits(stage_logits, output_size):
        resized = [
            F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
            for logits in stage_logits
        ]
        return torch.stack(resized, dim=0).mean(dim=0)

    def _prediction_prototypes(self, features, stage_logits):
        refined_fg, refined_bg = [], []
        for feat, logits in zip(features, stage_logits):
            prob = F.softmax(logits.detach(), dim=1)
            batch_size, channels, height, width = feat.shape
            feat_flat = feat.reshape(batch_size, channels, height * width)
            fg_flat = prob[:, 1].reshape(batch_size, height * width)
            bg_flat = prob[:, 0].reshape(batch_size, height * width)
            fg_items, bg_items = [], []
            for index in range(batch_size):
                fg_mask = fg_flat[index] > 0.7
                bg_mask = bg_flat[index] > 0.6
                if not bool(fg_mask.any()):
                    k = min(self.topk_fallback, fg_flat.shape[-1])
                    fg_mask = torch.zeros_like(fg_mask, dtype=torch.bool)
                    fg_mask[torch.topk(fg_flat[index], k).indices] = True
                if not bool(bg_mask.any()):
                    k = min(self.topk_fallback, bg_flat.shape[-1])
                    bg_mask = torch.zeros_like(bg_mask, dtype=torch.bool)
                    bg_mask[torch.topk(bg_flat[index], k).indices] = True
                fg_items.append(feat_flat[index, :, fg_mask].mean(dim=-1))
                bg_items.append(feat_flat[index, :, bg_mask].mean(dim=-1))
            refined_fg.append(torch.nan_to_num(torch.stack(fg_items, dim=0)))
            refined_bg.append(torch.nan_to_num(torch.stack(bg_items, dim=0)))
        return refined_fg, refined_bg

    @staticmethod
    def _blend_first(support_fg, support_bg, query_fg, query_bg, blend):
        blend = max(0.0, min(1.0, float(blend)))
        fg = [torch.nan_to_num((1.0 - blend) * sf + blend * qf) for sf, qf in zip(support_fg, query_fg)]
        bg = [torch.nan_to_num((1.0 - blend) * sb + blend * qb) for sb, qb in zip(support_bg, query_bg)]
        return fg, bg

    @staticmethod
    def _blend_second(support_fg, support_bg, prev_fg, prev_bg, query_fg, query_bg, blend):
        blend = max(0.0, min(1.0, float(blend)))
        fg = [
            torch.nan_to_num((1.0 - blend) * sf + blend * (0.4 * pf + 0.6 * qf))
            for sf, pf, qf in zip(support_fg, prev_fg, query_fg)
        ]
        bg = [
            torch.nan_to_num((1.0 - blend) * sb + blend * (0.4 * pb + 0.6 * qb))
            for sb, pb, qb in zip(support_bg, prev_bg, query_bg)
        ]
        return fg, bg

    def _support_logits(self, support_features, fg_shot_prototypes, bg_shot_prototypes, output_size):
        stage_logits = []
        for feat, fg_proto, bg_proto in zip(support_features, fg_shot_prototypes, bg_shot_prototypes):
            batch_size, k_shot, channels, height, width = feat.shape
            feat_flat = feat.reshape(batch_size * k_shot, channels, height, width)
            fg_flat = fg_proto.reshape(batch_size * k_shot, channels)
            bg_flat = bg_proto.reshape(batch_size * k_shot, channels)
            stage_logits.append(self._cosine_logits(feat_flat, fg_flat, bg_flat))
        return self._fuse_logits(stage_logits, output_size)

    def forward(self, support_images, support_masks, query_images, query_masks=None):
        """
        support_images: [B, K, 3, H, W]
        support_masks:  [B, K, 1, H, W] or [B, K, H, W]
        query_images:   [B, 3, H, W]
        query_masks:    optional [B, 1, H, W], used for CrackNex-style self loss
        """
        if support_images.ndim != 5:
            raise ValueError("support_images must be [B,K,3,H,W], got %s" % (support_images.shape,))
        if support_masks.ndim == 4:
            support_masks = support_masks.unsqueeze(2)
        support_masks = support_masks.float()
        batch_size, k_shot = support_images.shape[:2]
        output_size = query_images.shape[-2:]

        support_flat = support_images.reshape(batch_size * k_shot, *support_images.shape[2:])
        support_features_flat = self.encoder(support_flat)
        query_features = self.encoder(query_images)
        support_features = [
            feat.reshape(batch_size, k_shot, feat.shape[1], feat.shape[2], feat.shape[3])
            for feat in support_features_flat
        ]

        fg_raw, bg_raw, fg_shot, bg_shot, shot_valid = self._compute_support_prototypes(support_features, support_masks)
        fusion = self.prototype_fusion(fg_raw, bg_raw)
        fg_support = fusion["fused_fg_prototypes"]
        bg_support = fusion.get("fused_bg_prototypes", bg_raw)

        stage_logits0 = self._stage_logits(query_features, fg_support, bg_support)
        out0 = self._fuse_logits(stage_logits0, output_size)

        if self.use_ssp and self.ssp_blend > 0.0:
            q_fg1, q_bg1 = self._prediction_prototypes(query_features, stage_logits0)
            fg1, bg1 = self._blend_first(fg_support, bg_support, q_fg1, q_bg1, self.ssp_blend * 0.5)
            stage_logits1 = self._stage_logits(query_features, fg1, bg1)
            out1 = self._fuse_logits(stage_logits1, output_size)

            q_fg2, q_bg2 = self._prediction_prototypes(query_features, stage_logits1)
            fg2, bg2 = self._blend_second(fg_support, bg_support, fg1, bg1, q_fg2, q_bg2, self.ssp_blend * 0.5)
            stage_logits2 = self._stage_logits(query_features, fg2, bg2)
            out2 = self._fuse_logits(stage_logits2, output_size)
            fg_final, bg_final = fg2, bg2
        else:
            stage_logits1 = stage_logits0
            stage_logits2 = stage_logits0
            out1 = out0
            out2 = out0
            fg_final, bg_final = fg_support, bg_support

        self_logits = None
        if self.training and query_masks is not None:
            query_fg, query_bg = self._compute_mask_prototypes(query_features, query_masks)
            self_logits = self._fuse_logits(self._stage_logits(query_features, query_fg, query_bg), output_size)

        support_logits = None
        support_target = None
        if self.training:
            support_logits = self._support_logits(support_features, fg_shot, bg_shot, support_masks.shape[-2:])
            support_target = (support_masks.reshape(batch_size * k_shot, *support_masks.shape[-2:]) > 0.5).long()

        return {
            "logits": out2,
            "out0": out0,
            "out1": out1,
            "out2": out2,
            "aux_logits": [out1],
            "boundary_logits": None,
            "self_logits": self_logits,
            "support_logits": support_logits,
            "support_target": support_target,
            "fg_prototypes_raw": fg_raw,
            "bg_prototypes_raw": bg_raw,
            "fg_prototypes_orthogonal": fg_support,
            "fg_prototypes_refined": fg_final,
            "bg_prototypes_refined": bg_final,
            "similarity_maps_initial": [logits[:, 1:2] for logits in stage_logits0],
            "similarity_maps_final": [logits[:, 1:2] for logits in stage_logits2],
            "ssp_weights": [F.softmax(logits.detach(), dim=1)[:, 1:2] for logits in stage_logits2],
            "alpha": fusion.get("alpha"),
            "att_bg_vector": fusion.get("att_bg_vector"),
            "orthogonal_fg_vector": fusion.get("orthogonal_fg_vector"),
            "prototype_sep_loss": fusion.get("prototype_sep_loss"),
            "prototype_sep_similarity": fusion.get("prototype_sep_similarity"),
            "prototype_stage_sep_similarity": fusion.get("prototype_stage_sep_similarity"),
            "projected_bg_component": fusion.get("projected_bg_component"),
            "shot_valid": shot_valid,
        }

