import torch
import torch.nn as nn


class BinaryDiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, target):
        target = target.float()
        valid = target != 255
        prob = torch.sigmoid(logits.float())
        prob = prob.masked_select(valid)
        target = target.masked_select(valid)
        if prob.numel() == 0:
            return logits.sum() * 0.0
        inter = (prob * target).sum()
        denom = prob.sum() + target.sum()
        return 1.0 - (2.0 * inter + self.eps) / (denom + self.eps)

