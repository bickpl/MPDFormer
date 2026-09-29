from pathlib import Path

import numpy as np
from PIL import Image
import torch


def denormalize_image(tensor):
    if torch.is_tensor(tensor):
        image = tensor.detach().cpu().float()
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        image = (image * std + mean).clamp(0, 1)
        image = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return image
    return tensor


def save_prediction_triplet(path, image_tensor, mask_tensor, prob_tensor, threshold=0.5):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = denormalize_image(image_tensor)
    gt = mask_tensor.detach().cpu().squeeze().numpy() > 0.5
    pred = prob_tensor.detach().cpu().squeeze().numpy() > threshold
    gt_rgb = np.repeat(gt[:, :, None].astype(np.uint8) * 255, 3, axis=2)
    pred_rgb = np.repeat(pred[:, :, None].astype(np.uint8) * 255, 3, axis=2)
    canvas = np.concatenate([image, gt_rgb, pred_rgb], axis=1)
    Image.fromarray(canvas).save(path)


def save_mask(path, prob_tensor, threshold=0.5):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = (prob_tensor.detach().cpu().squeeze().numpy() > threshold).astype(np.uint8) * 255
    Image.fromarray(mask).save(path)

