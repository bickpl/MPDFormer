import logging
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def load_image(filename: Path) -> Image.Image:
    ext = filename.suffix.lower()
    if ext == ".npy":
        return Image.fromarray(np.load(filename))
    if ext in [".pt", ".pth"]:
        return Image.fromarray(torch.load(filename).numpy())
    return Image.open(filename)


class Crack500SegFormerDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        mask_dir: str,
        mode: str = "train",
        crop_size: int = 640,
        base_scale: float = 1.0,
        ratio_range: Tuple[float, float] = (0.5, 2.0),
        mask_suffix: str = "_mask",
        repeat: int = 50,
        pad_to_multiple: int = 32,
    ):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.mode = mode
        self.crop_size = crop_size
        self.base_scale = base_scale
        self.ratio_range = ratio_range
        self.mask_suffix = mask_suffix
        self.repeat = max(1, repeat)
        self.pad_to_multiple = pad_to_multiple
        self.image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        assert mode in {"train", "val"}
        assert crop_size > 0
        assert base_scale > 0

        self.ids = sorted(file.stem for file in self.images_dir.iterdir() if file.is_file() and not file.name.startswith("."))
        if not self.ids:
            raise RuntimeError("No input file found in %s" % self.images_dir)
        self.path_map = self._build_path_map()
        self.mask_values = [0, 1]

        logging.info(
            "Created SegFormer %s dataset from %s: %d images, crop=%d, base_scale=%.3f, ratio=%s, repeat=%d",
            self.mode,
            self.images_dir,
            len(self.ids),
            self.crop_size,
            self.base_scale,
            self.ratio_range,
            self.repeat if self.mode == "train" else 1,
        )

    def _build_path_map(self):
        path_map = {}
        for idx in self.ids:
            img_candidates = list(self.images_dir.glob("%s.*" % idx))
            mask_candidates = list(self.mask_dir.glob("%s%s.*" % (idx, self.mask_suffix)))
            if len(img_candidates) != 1:
                raise FileNotFoundError("Expected one image for %s, found %s" % (idx, img_candidates))
            if len(mask_candidates) != 1:
                raise FileNotFoundError("Expected one mask for %s, found %s" % (idx, mask_candidates))
            path_map[idx] = {"image": img_candidates[0], "mask": mask_candidates[0]}
        return path_map

    def __len__(self):
        return len(self.ids) * self.repeat if self.mode == "train" else len(self.ids)

    def _load_pair(self, idx):
        img = load_image(self.path_map[idx]["image"]).convert("RGB")
        mask = load_image(self.path_map[idx]["mask"])
        mask = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255)
        if img.size != mask.size:
            raise ValueError("Image and mask %s should have the same size, got %s and %s" % (idx, img.size, mask.size))
        return img, mask

    def _resize_pair(self, img, mask, scale):
        w, h = img.size
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = img.resize((new_w, new_h), resample=Image.BICUBIC)
        mask = mask.resize((new_w, new_h), resample=Image.NEAREST)
        return img, mask

    def _pad_to_crop(self, img, mask):
        w, h = img.size
        pad_w = max(0, self.crop_size - w)
        pad_h = max(0, self.crop_size - h)
        if pad_w or pad_h:
            img = ImageOps.expand(img, border=(0, 0, pad_w, pad_h), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, pad_w, pad_h), fill=0)
        return img, mask

    def _random_crop_pair(self, img, mask):
        img, mask = self._pad_to_crop(img, mask)
        w, h = img.size
        left = random.randint(0, w - self.crop_size)
        top = random.randint(0, h - self.crop_size)
        box = (left, top, left + self.crop_size, top + self.crop_size)
        return img.crop(box), mask.crop(box)

    @staticmethod
    def _random_flip_pair(img, mask):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask

    @staticmethod
    def _photometric_distort(img):
        if random.random() < 0.5:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
        if random.random() < 0.5:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
        if random.random() < 0.5:
            img = ImageEnhance.Color(img).enhance(random.uniform(0.8, 1.2))
        return img

    def _pad_to_multiple(self, image_tensor, mask_tensor):
        if self.pad_to_multiple <= 1:
            return image_tensor, mask_tensor
        h, w = mask_tensor.shape[-2:]
        pad_h = (self.pad_to_multiple - h % self.pad_to_multiple) % self.pad_to_multiple
        pad_w = (self.pad_to_multiple - w % self.pad_to_multiple) % self.pad_to_multiple
        if pad_h or pad_w:
            image_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h), value=0.0)
            mask_tensor = F.pad(mask_tensor, (0, pad_w, 0, pad_h), value=255)
        return image_tensor, mask_tensor

    def _to_tensor_pair(self, img, mask):
        image_tensor = self.image_transform(img)
        mask_tensor = torch.from_numpy((np.asarray(mask) > 0).astype(np.int64))
        if self.mode == "val":
            image_tensor, mask_tensor = self._pad_to_multiple(image_tensor, mask_tensor)
        return image_tensor, mask_tensor

    def __getitem__(self, sample_idx):
        idx = self.ids[sample_idx % len(self.ids)]
        img, mask = self._load_pair(idx)

        if self.mode == "train":
            scale = self.base_scale * random.uniform(*self.ratio_range)
            img, mask = self._resize_pair(img, mask, scale)
            img, mask = self._random_crop_pair(img, mask)
            img, mask = self._random_flip_pair(img, mask)
            img = self._photometric_distort(img)
        else:
            img, mask = self._resize_pair(img, mask, self.base_scale)

        image_tensor, mask_tensor = self._to_tensor_pair(img, mask)
        return {"image": image_tensor.float().contiguous(), "mask": mask_tensor.long().contiguous()}


class Crack500DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        batch_size: int = 2,
        num_workers: Optional[int] = None,
        img_scale: float = 1.0,
        crop_size: int = 640,
        ratio_min: float = 0.5,
        ratio_max: float = 2.0,
        repeat: int = 50,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = 0 if num_workers is None else num_workers
        self.img_scale = img_scale
        self.crop_size = crop_size
        self.ratio_range = (ratio_min, ratio_max)
        self.repeat = repeat
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and self.num_workers > 0
        self.prefetch_factor = prefetch_factor if self.num_workers > 0 else None
        self.train_set = None
        self.val_set = None
        self.mask_values = [0, 1]

    def setup(self, stage=None):
        self.train_set = Crack500SegFormerDataset(
            self.data_root / "train" / "image",
            self.data_root / "train" / "mask",
            mode="train",
            crop_size=self.crop_size,
            base_scale=self.img_scale,
            ratio_range=self.ratio_range,
            repeat=self.repeat,
        )
        self.val_set = Crack500SegFormerDataset(
            self.data_root / "val" / "image",
            self.data_root / "val" / "mask",
            mode="val",
            crop_size=self.crop_size,
            base_scale=self.img_scale,
            ratio_range=self.ratio_range,
            repeat=1,
        )

    def train_dataloader(self):
        kwargs = dict(
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=True,
        )
        if self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(self.train_set, **kwargs)

    def val_dataloader(self):
        kwargs = dict(
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )
        if self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(self.val_set, **kwargs)


Crack500SegmentationDataset = Crack500SegFormerDataset
