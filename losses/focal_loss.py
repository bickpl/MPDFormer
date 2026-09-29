import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target):
        target = target.float()
        valid = target != 255
        logits = logits.float()
        bce = F.binary_cross_entropy_with_logits(logits, target.clamp(0, 1), reduction="none")
        prob = torch.sigmoid(logits)
        pt = prob * target + (1.0 - prob) * (1.0 - target)
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        loss = alpha_t * (1.0 - pt).pow(self.gamma) * bce
        loss = loss.masked_select(valid)
        return loss.mean() if loss.numel() else logits.sum() * 0.0

