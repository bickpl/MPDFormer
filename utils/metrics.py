import numpy as np
import torch


def compute_binary_metrics_from_counts(tp, fp, fn, tn):
    eps = 1e-6
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)
    accuracy = (tp + tn + eps) / (tp + fp + fn + tn + eps)
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    bg_iou = (tn + eps) / (tn + fp + fn + eps)
    return {
        "iou": iou,
        "bg_iou": bg_iou,
        "miou": 0.5 * (iou + bg_iou),
        "dice": dice,
        "f1": dice,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "specificity": specificity,
        "mae": None,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def compute_binary_metrics(logits_or_prob, target, threshold=0.5, from_logits=True):
    if torch.is_tensor(logits_or_prob):
        if from_logits and logits_or_prob.ndim >= 4 and logits_or_prob.shape[1] == 2:
            prob = torch.softmax(logits_or_prob.float(), dim=1)[:, 1:2]
        else:
            prob = torch.sigmoid(logits_or_prob) if from_logits else logits_or_prob
        target = target.float()
        valid = target != 255
        pred = (prob > threshold) & valid
        gt = (target > 0.5) & valid
        tp = torch.logical_and(pred, gt).sum().item()
        fp = torch.logical_and(pred, ~gt & valid).sum().item()
        fn = torch.logical_and(~pred & valid, gt).sum().item()
        tn = torch.logical_and(~pred & valid, ~gt & valid).sum().item()
        metrics = compute_binary_metrics_from_counts(float(tp), float(fp), float(fn), float(tn))
        metrics["mae"] = torch.abs(prob.float() - gt.float()).masked_select(valid).mean().item() if valid.any() else 0.0
        return metrics

    prob = 1.0 / (1.0 + np.exp(-logits_or_prob)) if from_logits else logits_or_prob
    target = target.astype(np.float32)
    pred = prob > threshold
    gt = target > 0.5
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    tn = float(np.logical_and(~pred, ~gt).sum())
    metrics = compute_binary_metrics_from_counts(tp, fp, fn, tn)
    metrics["mae"] = float(np.abs(prob - gt.astype(np.float32)).mean())
    return metrics


class BinarySegMetrics:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0
        self.mae_sum = 0.0
        self.mae_count = 0

    def update(self, logits, target):
        with torch.no_grad():
            if logits.ndim >= 4 and logits.shape[1] == 2:
                prob = torch.softmax(logits.float(), dim=1)[:, 1:2]
            else:
                prob = torch.sigmoid(logits.float())
            target = target.float()
            valid = target != 255
            pred = (prob > self.threshold) & valid
            gt = (target > 0.5) & valid
            self.tp += float(torch.logical_and(pred, gt).sum().item())
            self.fp += float(torch.logical_and(pred, ~gt & valid).sum().item())
            self.fn += float(torch.logical_and(~pred & valid, gt).sum().item())
            self.tn += float(torch.logical_and(~pred & valid, ~gt & valid).sum().item())
            if valid.any():
                self.mae_sum += float(torch.abs(prob - gt.float()).masked_select(valid).sum().item())
                self.mae_count += int(valid.sum().item())

    def compute(self):
        metrics = compute_binary_metrics_from_counts(self.tp, self.fp, self.fn, self.tn)
        metrics["mae"] = self.mae_sum / max(1, self.mae_count)
        return metrics

