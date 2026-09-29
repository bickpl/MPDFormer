import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=padding, dilation=dilation, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class CrackDirectionalBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.branch_3x3 = DepthwiseSeparableConv(channels, 3, padding=1)
        self.branch_1x7 = DepthwiseSeparableConv(channels, (1, 7), padding=(0, 3))
        self.branch_7x1 = DepthwiseSeparableConv(channels, (7, 1), padding=(3, 0))
        self.branch_dilated = DepthwiseSeparableConv(channels, 3, padding=2, dilation=2)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        branches = [self.branch_3x3(x), self.branch_1x7(x), self.branch_7x1(x), self.branch_dilated(x)]
        return self.fuse(torch.cat(branches, dim=1))


class StageEnhanceBlock(nn.Module):
    def __init__(self, in_channels, decoder_dim):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Conv2d(in_channels, decoder_dim, 1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
        )
        self.prototype_proj = nn.Linear(in_channels, decoder_dim)
        self.enhance = nn.Sequential(
            nn.Conv2d(decoder_dim * 3 + 1, decoder_dim, 1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            CrackDirectionalBlock(decoder_dim),
        )

    def forward(self, feature, prototype, similarity):
        feat = self.feature_proj(feature)
        proto = self.prototype_proj(prototype)[:, :, None, None]
        proto = proto.expand_as(feat)
        if similarity.shape[-2:] != feat.shape[-2:]:
            similarity = F.interpolate(similarity, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        guided = torch.cat([feat, feat * proto, similarity, feat * torch.sigmoid(similarity)], dim=1)
        return self.enhance(guided)


class ACFPrototypeDecoder(nn.Module):
    def __init__(self, stage_channels, decoder_dim=256, use_aux=True, use_boundary=True, dropout=0.2):
        super().__init__()
        self.use_aux = use_aux
        self.use_boundary = use_boundary
        self.stage_blocks = nn.ModuleList([StageEnhanceBlock(ch, decoder_dim) for ch in stage_channels])
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * 4 + 4, decoder_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(float(dropout)),
            nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
        )
        self.boundary_head = nn.Conv2d(decoder_dim, 1, 1) if use_boundary else None
        self.seg_head = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_dim // 2, 1, 1),
        )
        self.aux_heads = nn.ModuleList([nn.Conv2d(decoder_dim, 1, 1) for _ in stage_channels]) if use_aux else None

    def forward(self, query_features, refined_prototypes, similarity_maps, output_size):
        enhanced = []
        for feat, proto, sim, block in zip(query_features, refined_prototypes, similarity_maps, self.stage_blocks):
            enhanced.append(block(feat, proto, sim))

        c1_size = enhanced[0].shape[-2:]
        enhanced_up = [enhanced[0]] + [
            F.interpolate(item, size=c1_size, mode="bilinear", align_corners=False) for item in enhanced[1:]
        ]
        sim_up = [similarity_maps[0]] + [
            F.interpolate(item, size=c1_size, mode="bilinear", align_corners=False) for item in similarity_maps[1:]
        ]
        fused = self.fuse(torch.cat(enhanced_up + sim_up, dim=1))

        boundary_logits = None
        if self.boundary_head is not None:
            boundary_logits = self.boundary_head(enhanced_up[0])
            fused = fused * (1.0 + torch.sigmoid(boundary_logits))

        logits = self.seg_head(fused)
        logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        if boundary_logits is not None:
            boundary_logits = F.interpolate(boundary_logits, size=output_size, mode="bilinear", align_corners=False)

        aux_logits = []
        if self.aux_heads is not None:
            for feat, head in zip(enhanced, self.aux_heads):
                aux = head(feat)
                aux_logits.append(F.interpolate(aux, size=output_size, mode="bilinear", align_corners=False))

        return {
            "logits": logits,
            "aux_logits": aux_logits,
            "boundary_logits": boundary_logits,
            "similarity_maps": similarity_maps,
        }

