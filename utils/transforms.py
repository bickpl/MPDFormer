import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FallbackTransform:
    def __init__(self, img_size=512, train=True, aug_mode="resize"):
        self.img_size = int(img_size)
        self.train = bool(train)
        self.aug_mode = aug_mode

    def __call__(self, image, mask):
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)
        if self.train and self.aug_mode in ("cracknex", "generalize"):
            w, h = image.size
            padw = max(0, self.img_size - w)
            padh = max(0, self.img_size - h)
            if padw > 0 or padh > 0:
                from PIL import ImageOps

                image = ImageOps.expand(image, border=(0, 0, padw, padh), fill=0)
                mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
            w, h = image.size
            x = np.random.randint(0, max(1, w - self.img_size + 1))
            y = np.random.randint(0, max(1, h - self.img_size + 1))
            image = image.crop((x, y, x + self.img_size, y + self.img_size))
            mask = mask.crop((x, y, x + self.img_size, y + self.img_size))
            if np.random.rand() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if self.aug_mode == "generalize":
                if np.random.rand() < 0.2:
                    image = image.transpose(Image.FLIP_TOP_BOTTOM)
                    mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
                if np.random.rand() < 0.5:
                    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.75, 1.25))
                    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.75, 1.30))
                if np.random.rand() < 0.15:
                    image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))
        else:
            image = image.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
            mask = mask.resize((self.img_size, self.img_size), resample=Image.NEAREST)
        image = np.asarray(image).astype(np.float32) / 255.0
        mask = (np.asarray(mask) > 0).astype(np.float32)
        image = (image - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(IMAGENET_STD, dtype=np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        return {"image": image, "mask": mask}


def build_transform(img_size=512, train=True, scale_limit=0.35, rotate_limit=20, aug_mode="cracknex"):
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except Exception:
        return FallbackTransform(img_size, train=train, aug_mode=aug_mode)

    if train:
        if aug_mode == "generalize":
            return A.Compose(
                [
                    A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, value=0, mask_value=0),
                    A.RandomCrop(img_size, img_size),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.2),
                    A.RandomRotate90(p=0.25),
                    A.ShiftScaleRotate(
                        shift_limit=0.06,
                        scale_limit=float(scale_limit),
                        rotate_limit=int(rotate_limit),
                        border_mode=0,
                        value=0,
                        mask_value=0,
                        p=0.65,
                    ),
                    A.OneOf(
                        [
                            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25),
                            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=18, val_shift_limit=16),
                            A.CLAHE(clip_limit=2.0),
                        ],
                        p=0.65,
                    ),
                    A.OneOf(
                        [
                            A.GaussianBlur(blur_limit=(3, 5)),
                            A.GaussNoise(var_limit=(5.0, 25.0)),
                            A.ImageCompression(quality_lower=65, quality_upper=95),
                        ],
                        p=0.25,
                    ),
                    A.CoarseDropout(
                        max_holes=4,
                        max_height=max(8, img_size // 16),
                        max_width=max(8, img_size // 16),
                        min_holes=1,
                        min_height=4,
                        min_width=4,
                        fill_value=0,
                        mask_fill_value=0,
                        p=0.15,
                    ),
                    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                    ToTensorV2(),
                ]
            )
        if aug_mode == "cracknex":
            return A.Compose(
                [
                    A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, value=0, mask_value=0),
                    A.RandomCrop(img_size, img_size),
                    A.HorizontalFlip(p=0.5),
                    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                    ToTensorV2(),
                ]
            )
        if aug_mode == "resize":
            return A.Compose(
                [
                    A.Resize(img_size, img_size),
                    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                    ToTensorV2(),
                ]
            )
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=float(scale_limit),
                    rotate_limit=int(rotate_limit),
                    border_mode=0,
                    p=0.6,
                ),
                A.RandomBrightnessContrast(p=0.5),
                A.CLAHE(p=0.2),
                A.GaussianBlur(blur_limit=(3, 5), p=0.15),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def apply_transform(transform, image, mask):
    result = transform(image=np.asarray(image), mask=np.asarray(mask))
    image_tensor = result["image"].float()
    mask_tensor = result["mask"]
    if not torch.is_tensor(mask_tensor):
        mask_tensor = torch.from_numpy(mask_tensor)
    mask_tensor = (mask_tensor.float() > 0).float()
    if mask_tensor.ndim == 2:
        mask_tensor = mask_tensor.unsqueeze(0)
    elif mask_tensor.ndim == 3 and mask_tensor.shape[-1] == 1:
        mask_tensor = mask_tensor.permute(2, 0, 1)
    elif mask_tensor.ndim == 3 and mask_tensor.shape[0] != 1:
        mask_tensor = mask_tensor[:1]
    return image_tensor.contiguous(), mask_tensor.contiguous()
