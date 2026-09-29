from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from utils.transforms import apply_transform, build_transform


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_image(path: Path):
    return Image.open(path).convert("RGB")


def binarize_mask_array(mask, mode="bright", threshold=127, foreground_value=255):
    mask = np.asarray(mask)
    if mode == "nonzero":
        foreground = mask > 0
    elif mode == "bright":
        foreground = mask > int(threshold)
    elif mode == "dark":
        foreground = mask < int(threshold)
    elif mode == "auto":
        bright = mask > int(threshold)
        dark = mask < int(threshold)
        bright_ratio = float(bright.mean())
        dark_ratio = float(dark.mean())
        if bright_ratio == 0.0:
            foreground = bright
        elif dark_ratio == 0.0:
            foreground = dark
        elif bright_ratio <= 0.5 and (dark_ratio > 0.5 or bright_ratio <= dark_ratio):
            foreground = bright
        elif dark_ratio <= 0.5:
            foreground = dark
        else:
            foreground = bright if bright_ratio <= dark_ratio else dark
    else:
        raise ValueError("Unsupported mask mode %r; use bright, dark, nonzero, or auto" % mode)
    return foreground.astype(np.uint8) * int(foreground_value)


def load_mask(path: Path, mode="bright", threshold=127, foreground_value=255):
    mask = Image.open(path).convert("L")
    return Image.fromarray(binarize_mask_array(mask, mode=mode, threshold=threshold, foreground_value=foreground_value))


def build_path_map(images_dir, masks_dir):
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    if not images_dir.exists():
        raise FileNotFoundError("Images directory not found: %s" % images_dir)
    if not masks_dir.exists():
        raise FileNotFoundError("Masks directory not found: %s" % masks_dir)

    image_by_stem = {}
    for file in images_dir.iterdir():
        if file.is_file() and file.suffix.lower() in IMAGE_EXTS and not file.name.startswith("."):
            image_by_stem.setdefault(file.stem, []).append(file)
    mask_by_stem = {}
    for file in masks_dir.iterdir():
        if file.is_file() and file.suffix.lower() in IMAGE_EXTS and not file.name.startswith("."):
            mask_by_stem.setdefault(file.stem, []).append(file)

    path_map = {}
    for stem in sorted(image_by_stem):
        images = image_by_stem.get(stem, [])
        masks = mask_by_stem.get(stem, [])
        if len(images) != 1:
            raise FileNotFoundError("Expected one image for %s, found %s" % (stem, images))
        if len(masks) != 1:
            raise FileNotFoundError("Expected one mask for %s, found %s" % (stem, masks))
        path_map[stem] = {"image": images[0], "mask": masks[0]}
    if not path_map:
        raise RuntimeError("No image/mask pairs found under %s and %s" % (images_dir, masks_dir))
    return path_map


class CrackImageMaskDataset(Dataset):
    def __init__(
        self,
        images_dir,
        masks_dir,
        img_size=512,
        train=False,
        mask_mode="bright",
        mask_threshold=127,
        foreground_value=255,
    ):
        self.path_map = build_path_map(images_dir, masks_dir)
        self.ids = sorted(self.path_map)
        self.transform = build_transform(img_size, train=train)
        self.mask_mode = mask_mode
        self.mask_threshold = int(mask_threshold)
        self.foreground_value = int(foreground_value)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        name = self.ids[index]
        item = self.path_map[name]
        image = load_image(item["image"])
        mask = load_mask(
            item["mask"],
            mode=self.mask_mode,
            threshold=self.mask_threshold,
            foreground_value=self.foreground_value,
        )
        if image.size != mask.size:
            mask = mask.resize(image.size, resample=Image.NEAREST)
        image_tensor, mask_tensor = apply_transform(self.transform, image, mask)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "name": name,
            "image_path": str(item["image"]),
            "mask_path": str(item["mask"]),
        }


def mask_foreground_ratio(mask_path, mode="bright", threshold=127, foreground_value=255):
    mask = load_mask(Path(mask_path), mode=mode, threshold=threshold, foreground_value=foreground_value)
    return float((np.asarray(mask) > 0).mean())


def mask_has_foreground(
    mask_path,
    min_fg_ratio=0.0,
    max_fg_ratio=1.0,
    mode="bright",
    threshold=127,
    foreground_value=255,
):
    ratio = mask_foreground_ratio(
        mask_path,
        mode=mode,
        threshold=threshold,
        foreground_value=foreground_value,
    )
    return ratio > float(min_fg_ratio) and ratio <= float(max_fg_ratio)

