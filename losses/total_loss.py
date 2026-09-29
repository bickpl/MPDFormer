import torch
import torch.nn as nn
import torch.nn.functional as F


class MPDFormerLoss(nn.Module):
    def __init__(
        self,
        lambda_bce=1.0,
        lambda_dice=1.0,
        lambda_focal=0.25,
        focal_alpha=0.25,
        focal_gamma=2.0,
        proto_loss_weight=0.1,
    ):
        super().__init__()
        lambda_bce = float(lambda_bce)
        lambda_dice = float(lambda_dice)
        lambda_focal = float(lambda_focal)
        lambda_sum = lambda_bce + lambda_dice + lambda_focal
        if lambda_sum <= 0:
            raise ValueError("lambda_bce + lambda_dice + lambda_focal must be positive")

        initial_lambdas = torch.tensor([lambda_bce, lambda_dice, lambda_focal], dtype=torch.float32)
        self.register_buffer("lambda_sum", torch.tensor(float(lambda_sum), dtype=torch.float32))
        self.raw_seg_lambdas = nn.Parameter(torch.log(initial_lambdas / float(lambda_sum)).float())

        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.proto_loss_weight = float(proto_loss_weight)

        # 双通道用 CrossEntropyLoss，支持ignore_index=255
        self.ce = nn.CrossEntropyLoss(reduction="mean", ignore_index=255)

    def learned_lambdas(self):
        weights = F.softmax(self.raw_seg_lambdas, dim=0) * self.lambda_sum.to(self.raw_seg_lambdas.device)
        return {
            "lambda_bce": weights[0],
            "lambda_dice": weights[1],
            "lambda_focal": weights[2],
        }

    @staticmethod
    def _target_to_long(target):
        """ target [B,1,H,W] mask:0=bg,1=crack,255=ignore """
        if target.ndim == 4 and target.shape[1] == 1:
            target = target[:, 0]  # [B,H,W]
        ignore = target == 255
        target_long = (target.float() > 0.5).long()
        return target_long.masked_fill(ignore, 255)

    @staticmethod
    def _valid_mask(target_long):
        return target_long != 255

    def _ce_loss(self, logits_2ch, target_long):
        """
        logits_2ch: [B,2,H,W] 双通道原始logits
        target_long: [B,H,W] value 0/1/255
        CrossEntropyLoss原生支持ignore_index=255，不需要手动mask_select
        """
        return self.ce(logits_2ch, target_long)

    def _dice_loss(self, logits_2ch, target_long):
        """取裂纹通道 index=1，做dice"""
        valid = self._valid_mask(target_long)
        if not bool(valid.any()):
            return logits_2ch.sum() * 0.0
        # softmax得到概率，取裂纹通道
        prob = F.softmax(logits_2ch.float(), dim=1)[:, 1, :, :]  # [B,H,W]
        target = (target_long == 1).float()

        prob_valid = torch.masked_select(prob, valid)
        target_valid = torch.masked_select(target, valid)

        inter = 2.0 * (prob_valid * target_valid).sum()
        denom = prob_valid.sum() + target_valid.sum()
        eps = logits_2ch.new_tensor(1e-6)
        return 1.0 - (inter + eps) / (denom + eps)

    def _focal_loss(self, logits_2ch, target_long):
        valid = self._valid_mask(target_long)
        if not bool(valid.any()):
            return logits_2ch.sum() * 0.0

        B, _, H, W = logits_2ch.shape
        # cross‑entropy focal，双通道版本
        logp = F.log_softmax(logits_2ch, dim=1)
        # 取出对应类别的log概率
        logpt = torch.gather(logp, dim=1, index=target_long.unsqueeze(1))[:, 0]  # [B,H,W]

        pt = torch.exp(logpt)
        alpha_t = torch.where(
            target_long == 1,
            torch.full_like(pt, self.focal_alpha),
            torch.full_like(pt, 1.0 - self.focal_alpha)
        )
        # focal: -α_t * (1‑pt)^γ * log(pt)
        focal_all = - alpha_t * torch.pow((1.0 - pt), self.focal_gamma) * logpt

        focal_valid = torch.masked_select(focal_all, valid)
        return focal_valid.mean()

    def _seg_loss(self, logits_2ch, target):
        """
        logits_2ch: [B,2,H,W]
        target: [B,1,H,W] mask tensor
        """
        target_long = self._target_to_long(target)  # [B,H,W] 0/1/255

        loss_ce = self._ce_loss(logits_2ch, target_long)
        loss_dice = self._dice_loss(logits_2ch, target_long)
        loss_focal = self._focal_loss(logits_2ch, target_long)

        lambdas = self.learned_lambdas()
        seg_total = (
            lambdas["lambda_bce"] * loss_ce
            + lambdas["lambda_dice"] * loss_dice
            + lambdas["lambda_focal"] * loss_focal
        )
        return seg_total, loss_ce, loss_dice, loss_focal

    def forward(self, outputs, target):
        """
        outputs: dict, must contain "logits" -> [B,2,H,W]
        target: [B,1,H,W] mask tensor
        """
        logits_2ch = outputs["logits"]
        seg_total, loss_ce, loss_dice, loss_focal = self._seg_loss(logits_2ch, target)

        total_loss = seg_total

        loss_parts = {
            "bce": loss_ce,
            "dice": loss_dice,
            "focal": loss_focal,
            "total": total_loss
        }
        return total_loss, loss_parts
