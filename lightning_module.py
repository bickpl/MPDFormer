import copy
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from losses import MPDFormerLoss
from models import PrototypeACFSegFormer


class LitMPDFormer(pl.LightningModule):
    def __init__(
        self,
        backbone: str = "b3",
        pretrained_backbone: bool = True,
        backbone_weights_path: str = "",
        freeze_encoder: bool = False,
        freeze_encoder_stages: int = 0,
        decoder_dim: int = 256,
        temperature: float = 10.0,
        ssp_mode: str = "soft",
        ssp_top_ratio: float = 0.15,
        use_ssp: bool = True,
        ssp_blend: float = 0.5,
        alpha_min: float = 0.05,
        alpha_max: float = 0.95,
        sep_margin: float = 0.1,
        decoder_dropout: float = 0.2,
        use_aux: bool = True,
        use_boundary: bool = True,
        lambda_bce: float = 1.0,
        lambda_dice: float = 1.0,
        lambda_focal: float = 0.25,

        freeze_batchnorm: bool = False,
        use_ema: bool = False,
        lr = 1.0e-4,
        encoder_lr_mult = 0.5,
        optimizer: str = "adamw",
        scheduler: str = "cosine",
        momentum: float = 0.9,
        lr_milestone_gamma: float = 0.5,
        max_epochs: int = 200,
        warmup_epochs: int = 5,
        weight_decay: float = 0.01,
        pred_threshold: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = PrototypeACFSegFormer(
            backbone=backbone,
            pretrained_backbone=pretrained_backbone,
            backbone_weights_path=backbone_weights_path,
            decoder_dim=decoder_dim,
            temperature=temperature,
            freeze_encoder=freeze_encoder,
            freeze_encoder_stages=freeze_encoder_stages,
            ssp_mode=ssp_mode,
            ssp_top_ratio=ssp_top_ratio,
            use_ssp=use_ssp,
            ssp_blend=ssp_blend,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            sep_margin=sep_margin,
            decoder_dropout=decoder_dropout,
            use_aux=use_aux,
            use_boundary=use_boundary,
        )
        self.criterion = MPDFormerLoss(
            lambda_bce=lambda_bce,
            lambda_dice=lambda_dice,
            lambda_focal=lambda_focal,
        )
        self.ema_model = copy.deepcopy(self.model) if use_ema else None
        if self.ema_model is not None:
            for param in self.ema_model.parameters():
                param.requires_grad_(False)
        if freeze_batchnorm:
            self._freeze_batchnorm_layers()
        self.mask_values = [0, 1]
        self.example_input_array = None
        self._val_vis_sample = None
        self._val_vis_seen = 0
        self.register_buffer("val_sweep_thresholds", torch.linspace(0.1, 0.9, 17), persistent=False)
        self._val_sweep_tp = None
        self._val_sweep_fp = None
        self._val_sweep_fn = None

    @property
    def n_channels(self):
        return 3

    @property
    def n_classes(self):
        return 2

    def forward(self, support_images, support_masks, query_image, query_mask=None):
        return self.model(support_images.float(), support_masks.float(), query_image.float(), query_mask)

    def _freeze_batchnorm_layers(self):
        for module in self.model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
                for param in module.parameters():
                    param.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if mode and bool(self.hparams.freeze_batchnorm):
            self._freeze_batchnorm_layers()
        return self

    def _forward_for_step(self, support_images, support_masks, query_image, query_mask=None):
        model = self.ema_model if (not self.training and self.ema_model is not None) else self.model
        return model(support_images.float(), support_masks.float(), query_image.float(), query_mask)

    def _shared_step(self, batch):
        support_images = batch["support_images"]
        support_masks = batch["support_masks"]
        query_image = batch["query_image"]
        query_mask = batch["query_mask"].float()
        outputs = self._forward_for_step(support_images, support_masks, query_image, query_mask)

        loss, loss_parts = self.criterion(outputs, query_mask)
        if not torch.isfinite(loss):
            raise FloatingPointError("NaN/Inf loss detected in MPDFormer batch")
        prob = F.softmax(outputs["logits"].float(), dim=1)[:, 1:2]
        valid = query_mask != 255
        target = (query_mask > 0.5).float()
        metrics = self._binary_metrics(prob, target, valid)
        return outputs, loss, loss_parts, prob, target, valid, metrics

    @staticmethod
    def _soft_dice(prob, target, valid):
        valid = valid.float()
        prob = prob.float() * valid
        target = target.float() * valid
        eps = torch.tensor(1e-6, device=prob.device, dtype=prob.dtype)
        inter = 2.0 * (prob * target).sum()
        denom = prob.sum() + target.sum()
        return (inter + eps) / (denom + eps)

    def _binary_metrics(self, prob, target, valid):
        prob = prob.float()
        target = target.float()
        valid = valid.float()
        pred = (prob > float(self.hparams.pred_threshold)).float() * valid
        target = target * valid
        inv_pred = (1.0 - pred) * valid
        inv_target = (1.0 - target) * valid

        tp = (pred * target).sum()
        fp = (pred * inv_target).sum()
        fn = (inv_pred * target).sum()
        tn = (inv_pred * inv_target).sum()
        eps = torch.tensor(1e-6, device=prob.device, dtype=prob.dtype)
        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)
        specificity = (tn + eps) / (tn + fp + eps)
        iou = (tp + eps) / (tp + fp + fn + eps)
        bg_iou = (tn + eps) / (tn + fp + fn + eps)
        dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
        valid_count = valid.sum().clamp_min(1.0)
        return {
            "dice": dice,
            "soft_dice": self._soft_dice(prob, target, valid),
            "iou": iou,
            "bg_iou": bg_iou,
            "miou": 0.5 * (iou + bg_iou),
            "precision": precision,
            "recall": recall,
            "f1": (2.0 * precision * recall + eps) / (precision + recall + eps),
            "accuracy": (tp + tn + eps) / (tp + fp + fn + tn + eps),
            "specificity": specificity,
            "mae": torch.abs(prob - target).mul(valid).sum() / valid_count,
            "gt_fg_ratio": target.sum() / valid_count,
            "pred_fg_ratio": pred.sum() / valid_count,
        }

    def _log_loss_and_metrics(self, prefix, loss, loss_parts, metrics, batch_size, prog_bar=False):
        self.log("%s_loss" % prefix, loss, on_step=False, on_epoch=True, prog_bar=prog_bar, batch_size=batch_size)
        loss_log = {
            "%s_main_loss" % prefix: loss_parts["total"],
            "%s_ce_loss" % prefix: loss_parts["bce"],
            "%s_dice_loss" % prefix: loss_parts["dice"],
            "%s_focal_loss" % prefix: loss_parts["focal"],
        }
        self.log_dict(loss_log, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)
        self.log("%s_dice" % prefix, metrics["dice"], on_step=False, on_epoch=True, prog_bar=prog_bar, batch_size=batch_size)
        self.log_dict(
            {
                "%s_soft_dice" % prefix: metrics["soft_dice"],
                "%s_iou" % prefix: metrics["iou"],
                "%s_bg_iou" % prefix: metrics["bg_iou"],
                "%s_miou" % prefix: metrics["miou"],
                "%s_precision" % prefix: metrics["precision"],
                "%s_recall" % prefix: metrics["recall"],
                "%s_f1" % prefix: metrics["f1"],
                "%s_accuracy" % prefix: metrics["accuracy"],
                "%s_specificity" % prefix: metrics["specificity"],
                "%s_mae" % prefix: metrics["mae"],
                "%s_gt_fg_ratio" % prefix: metrics["gt_fg_ratio"],
                "%s_pred_fg_ratio" % prefix: metrics["pred_fg_ratio"],
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )

    def training_step(self, batch, batch_idx):
        _, loss, loss_parts, _, _, _, metrics = self._shared_step(batch)
        self._log_loss_and_metrics("train", loss, loss_parts, metrics, batch["query_image"].shape[0], prog_bar=True)
        return loss

    @torch.no_grad()
    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ema_model is None:
            return
        decay = float(self.hparams.ema_decay)
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)
        for ema_buffer, buffer in zip(self.ema_model.buffers(), self.model.buffers()):
            ema_buffer.copy_(buffer)

    def on_validation_epoch_start(self):
        if self.trainer is not None and self.trainer.sanity_checking:
            return
        self._val_vis_sample = None
        self._val_vis_seen = 0
        device = self.device
        count = int(self.val_sweep_thresholds.numel())
        self._val_sweep_tp = torch.zeros(count, device=device)
        self._val_sweep_fp = torch.zeros(count, device=device)
        self._val_sweep_fn = torch.zeros(count, device=device)

    @torch.no_grad()
    def _update_val_threshold_sweep(self, prob, target, valid):
        if self._val_sweep_tp is None:
            return
        prob = prob.detach().float()
        target = target.detach().float()
        valid = valid.detach().bool()
        if not valid.any():
            return
        prob_flat = prob[valid].flatten()
        target_flat = (target[valid].flatten() > 0.5)
        thresholds = self.val_sweep_thresholds.to(prob_flat.device)
        pred = prob_flat.unsqueeze(0) > thresholds[:, None]
        gt = target_flat.unsqueeze(0)
        self._val_sweep_tp += torch.logical_and(pred, gt).sum(dim=1).float()
        self._val_sweep_fp += torch.logical_and(pred, ~gt).sum(dim=1).float()
        self._val_sweep_fn += torch.logical_and(~pred, gt).sum(dim=1).float()

    @staticmethod
    def _denorm_image_to_uint8(image_tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image_tensor.detach().cpu() * std + mean
        image = image.clamp(0, 1).numpy()
        image = np.transpose(image, (1, 2, 0))
        return (image * 255.0).astype(np.uint8)

    @staticmethod
    def _mask_to_uint8(mask_tensor):
        mask = mask_tensor.detach().cpu().numpy()
        mask = np.squeeze(mask)
        return (mask > 0).astype(np.uint8) * 255

    def _maybe_store_random_val_sample(self, batch, pred):
        batch_size = batch["query_image"].shape[0]
        for i in range(batch_size):
            self._val_vis_seen += 1
            if random.randint(1, self._val_vis_seen) != 1:
                continue
            self._val_vis_sample = {
                "image": batch["query_image"][i].detach().cpu(),
                "mask": batch["query_mask"][i].detach().cpu(),
                "pred": pred[i].detach().cpu(),
            }

    def validation_step(self, batch, batch_idx):
        outputs, loss, loss_parts, prob, target, valid, metrics = self._shared_step(batch)
        pred = (prob > float(self.hparams.pred_threshold)).long()
        if self.trainer is None or not self.trainer.sanity_checking:
            self._update_val_threshold_sweep(prob, target, valid)
        self._maybe_store_random_val_sample(batch, pred)
        self._log_loss_and_metrics("val", loss, loss_parts, metrics, batch["query_image"].shape[0], prog_bar=True)
        return {"val_loss": loss, "val_dice": metrics["dice"], "val_iou": metrics["iou"], "val_miou": metrics["miou"]}

    def on_validation_epoch_end(self):
        if self._val_sweep_tp is not None:
            eps = torch.tensor(1e-6, device=self._val_sweep_tp.device)
            dice = (2.0 * self._val_sweep_tp + eps) / (
                2.0 * self._val_sweep_tp + self._val_sweep_fp + self._val_sweep_fn + eps
            )
            best_idx = torch.argmax(dice)
            self.log("val_best_dice", dice[best_idx], on_step=False, on_epoch=True, prog_bar=True)
            self.log("val_best_threshold", self.val_sweep_thresholds[best_idx], on_step=False, on_epoch=True, prog_bar=False)
        if self._val_vis_sample is None or self.trainer is None or self.trainer.sanity_checking:
            return
        image = self._denorm_image_to_uint8(self._val_vis_sample["image"])
        mask = self._mask_to_uint8(self._val_vis_sample["mask"])
        pred = self._mask_to_uint8(self._val_vis_sample["pred"])
        mask_rgb = np.stack([mask] * 3, axis=-1)
        pred_rgb = np.stack([pred] * 3, axis=-1)
        combined = np.concatenate([image, mask_rgb, pred_rgb], axis=1)

        save_dir = Path(self.trainer.default_root_dir) / "val_samples"
        save_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(combined).save(save_dir / ("epoch_%04d.png" % (self.current_epoch + 1)))


    def configure_optimizers(self):
        encoder_params = []
        head_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad or name.startswith("ema_model."):
                continue
            if name.startswith("model.encoder."):
                encoder_params.append(param)
            else:
                head_params.append(param)

        param_groups = []
        if encoder_params:
            param_groups.append(
                {
                    "params": encoder_params,
                    # ① learning_rate → lr
                    "lr": self.hparams.lr * float(self.hparams.encoder_lr_mult),
                    "name": "encoder",
                }
            )
        if head_params:
            param_groups.append({"params": head_params, "lr": self.hparams.lr, "name": "head"})

        # ② optimizer_name → optimizer，与yaml key对齐
        if self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                param_groups,
                lr=self.hparams.lr,  # learning_rate → lr
                momentum=float(self.hparams.momentum),
                weight_decay=self.hparams.weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=self.hparams.lr,  # learning_rate → lr
                weight_decay=self.hparams.weight_decay,
            )

        try:
            total_steps = int(self.trainer.estimated_stepping_batches)
        except Exception:
            total_steps = 1
        total_steps = max(1, total_steps)

        warmup_steps = max(0, int(self.hparams.warmup_epochs) * max(1, math.ceil(
            total_steps / max(1, int(self.hparams.max_epochs)))))
        step_milestones = {max(1, total_steps // 3), max(1, (2 * total_steps) // 3)}

        def lr_lambda(step):
            # ③ scheduler_name → scheduler，与yaml key对齐
            if self.hparams.scheduler == "step":
                factor = 1.0
                for milestone in step_milestones:
                    if step >= milestone:
                        factor *= float(self.hparams.lr_milestone_gamma)
                return factor

            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)

            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def on_save_checkpoint(self, checkpoint):
        checkpoint["mask_values"] = self.mask_values

    def on_load_checkpoint(self, checkpoint):
        if self.ema_model is None or not isinstance(checkpoint, dict):
            return
        state_dict = checkpoint.get("state_dict", {})
        if not any(key.startswith("ema_model.") for key in state_dict):
            self.ema_model.load_state_dict(self.model.state_dict(), strict=False)

