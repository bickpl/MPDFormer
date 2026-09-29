import torch
import torch.nn.functional as F


def masked_average_pooling(feature, mask, eps=1e-6):
    """
    feature: [B, C, H, W]
    mask:    [B, 1, Hm, Wm] or [B, 1, H, W]
    return:  [B, C]
    """
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[-2:] != feature.shape[-2:]:
        mask = F.interpolate(mask.float(), size=feature.shape[-2:], mode="nearest")
    mask = (mask.float() > 0.5).float()
    denom = mask.sum(dim=(2, 3)).clamp_min(eps)
    proto = (feature * mask).sum(dim=(2, 3)) / denom
    return torch.nan_to_num(proto)


def compute_multi_scale_prototypes(support_features, support_masks):
    """
    support_features[l]: [B, K, C_l, H_l, W_l]
    support_masks:       [B, K, 1, H, W]
    returns fg/bg lists: [B, C_l] for each stage.
    """
    if support_masks.ndim == 4:
        support_masks = support_masks.unsqueeze(2)
    batch_size, k_shot = support_masks.shape[:2]
    masks_flat = support_masks.reshape(batch_size * k_shot, 1, *support_masks.shape[-2:])
    shot_valid = (masks_flat.flatten(1).sum(dim=1) > 0).reshape(batch_size, k_shot, 1).float()
    fg_prototypes, bg_prototypes = [], []
    for feat in support_features:
        channels = feat.shape[2]
        feat_flat = feat.reshape(batch_size * k_shot, channels, *feat.shape[-2:])
        fg = masked_average_pooling(feat_flat, masks_flat).reshape(batch_size, k_shot, channels)
        bg = masked_average_pooling(feat_flat, 1.0 - masks_flat).reshape(batch_size, k_shot, channels).mean(dim=1)
        denom = shot_valid.sum(dim=1).clamp_min(1.0)
        fg_prototypes.append(torch.nan_to_num((fg * shot_valid).sum(dim=1) / denom))
        bg_prototypes.append(torch.nan_to_num(bg))
    return fg_prototypes, bg_prototypes


def cosine_similarity_map(feature, prototype, temperature=1.0):
    feature = F.normalize(feature, p=2, dim=1, eps=1e-6)
    prototype = F.normalize(prototype, p=2, dim=1, eps=1e-6)
    sim = (feature * prototype[:, :, None, None]).sum(dim=1, keepdim=True)
    return torch.nan_to_num(sim * float(temperature))

