import torch
import torch.nn as nn
import torch.nn.functional as F

from .dice_loss import BinaryDiceLoss


def mask_to_boundary(mask, kernel_size=3):
    mask = (mask.float() > 0.5).float()
    pad = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size, stride=1, padding=pad)
    eroded = -F.max_pool2d(-mask, kernel_size, stride=1, padding=pad)
    return (dilated - eroded).clamp(0, 1)


class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = BinaryDiceLoss()

    def forward(self, boundary_logits, mask):
        if boundary_logits is None:
            return mask.sum() * 0.0
        target = mask_to_boundary(mask)
        return self.bce(boundary_logits, target) + self.dice(boundary_logits, target)

