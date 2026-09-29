import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.net = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class OrthogonalPrototypeFusion(nn.Module):
    """
    Compatibility name for the MPDFormer prototype fusion block.


    """

    def __init__(
        self,
        stage_channels,
        reduction=8,
        eps=1e-6,
        alpha_min=0.05,
        alpha_max=0.95,
        sep_margin=0.1,
    ):
        super().__init__()
        self.stage_channels = list(stage_channels)
        self.total_channels = sum(self.stage_channels)
        self.eps = eps
        self.sep_margin = float(sep_margin)
        self.fg_attention = ChannelAttention(self.total_channels, reduction)
        self.bg_attention = ChannelAttention(self.total_channels, reduction)
        self.fg_norm = nn.LayerNorm(self.total_channels)
        self.bg_norm = nn.LayerNorm(self.total_channels)

    def _stage_separability_loss(self, fg_prototypes, bg_prototypes):
        losses = []
        similarities = []
        for fg, bg in zip(fg_prototypes, bg_prototypes):
            fg = F.normalize(fg, p=2, dim=1, eps=self.eps)
            bg = F.normalize(bg, p=2, dim=1, eps=self.eps)
            sim = (fg * bg).sum(dim=1)
            similarities.append(sim)
            losses.append(F.relu(sim - self.sep_margin))
        return torch.stack(losses, dim=0).mean(), torch.stack(similarities, dim=0).mean(dim=0)

    def forward(self, fg_prototypes, bg_prototypes):
        # Step1: Concatenate multi-scale prototypes along channel dimension
        fg_vector = torch.cat(fg_prototypes, dim=1)
        bg_vector = torch.cat(bg_prototypes, dim=1)

        # Step2: Channel Attention Module (CAM) to get weighted prototypes
        fg_att = fg_vector * self.fg_attention(fg_vector)
        bg_att = bg_vector * self.bg_attention(bg_vector)

        # Step3: Selective Orthogonal Projection
        inner_prod = torch.sum(fg_att * bg_att, dim=1, keepdim=True)
        bg_l2_sq = torch.sum(bg_att ** 2, dim=1, keepdim=True) + self.eps
        pos_coeff = torch.maximum(inner_prod, torch.zeros_like(inner_prod)) / bg_l2_sq

        proj_bg_component = pos_coeff * bg_att
        orth_fg = fg_att - proj_bg_component

        # Step4: L2 normalize for similarity calculation
        fg_clean = F.normalize(orth_fg, p=2, dim=1, eps=self.eps)
        bg_clean = F.normalize(bg_att, p=2, dim=1, eps=self.eps)

        # Step5: Global & stage separability loss
        global_sim = (fg_clean * bg_clean).sum(dim=1)
        global_loss = F.relu(global_sim - self.sep_margin).mean()

        # Split back to multi-scale prototypes
        fused_fg = list(torch.split(fg_clean, self.stage_channels, dim=1))
        fused_bg = list(torch.split(bg_clean, self.stage_channels, dim=1))
        stage_loss, stage_sim = self._stage_separability_loss(fused_fg, fused_bg)
        separability_loss = 0.5 * global_loss + 0.5 * stage_loss

        return {
            "fused_fg_prototypes": fused_fg,
            "fused_bg_prototypes": fused_bg,
            "att_fg_vector": fg_att,
            "att_bg_vector": bg_att,
            "fg_vector": fg_clean,
            "bg_vector": bg_clean,
            "orthogonal_fg_vector": orth_fg,
            "prototype_sep_loss": separability_loss,
            "prototype_sep_similarity": global_sim.detach(),
            "prototype_stage_sep_similarity": stage_sim.detach(),
            "projected_bg_component": proj_bg_component.detach(),
            "alpha": pos_coeff.detach(),
        }

