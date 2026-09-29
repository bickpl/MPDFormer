import random
from pathlib import Path
from typing import Optional

from PIL import Image
import torch
from torch.utils.data import Dataset

from .crack_dataset import build_path_map, load_image, load_mask, mask_has_foreground
from utils.transforms import apply_transform, build_transform


def split_dirs(data_root, split):
    root = Path(data_root)
    return root / split / "images", root / split / "masks"


def support_dirs(data_root, support_root=None):
    if support_root:
        root = Path(support_root)
        return root / "images", root / "masks"
    return split_dirs(data_root, "train")


def infer_sample_domain(name):
    for sep in ("_", "-"):
        if sep in name:
            return name.split(sep, 1)[0]
    return ""


class EpisodeCrackDataset(Dataset):
    def __init__(
        self,
        data_root,
        split="train",
        img_size=512,
        k_shot=1,
        episodes_per_epoch=1000,
        support_root: Optional[str] = None,
        allow_empty_support=False,
        mask_mode="bright",
        mask_threshold=127,
        foreground_value=255,
        support_mask_mode=None,
        support_mask_threshold=None,
        support_foreground_value=None,
        min_support_fg_ratio=0.001,
        max_support_fg_ratio=1.0,
        min_query_fg_ratio=0.001,
        max_query_fg_ratio=1.0,
        positive_query_prob=0.7,
        cross_domain_support_prob=0.3,
        crop_attempts=50,
        aug_mode="cracknex",
        aug_scale_limit=0.25,
        aug_rotate_limit=10,
        train=True,
        fixed_support=False,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = int(img_size)
        self.k_shot = int(k_shot)
        self.episodes_per_epoch = int(episodes_per_epoch)
        self.allow_empty_support = bool(allow_empty_support)
        self.mask_mode = mask_mode
        self.mask_threshold = int(mask_threshold)
        self.foreground_value = int(foreground_value)
        self.support_mask_mode = support_mask_mode or mask_mode
        self.support_mask_threshold = int(mask_threshold if support_mask_threshold is None else support_mask_threshold)
        self.support_foreground_value = int(foreground_value if support_foreground_value is None else support_foreground_value)
        self.min_support_fg_ratio = float(min_support_fg_ratio)
        self.max_support_fg_ratio = float(max_support_fg_ratio)
        self.min_query_fg_ratio = float(min_query_fg_ratio)
        self.max_query_fg_ratio = float(max_query_fg_ratio)
        self.positive_query_prob = float(positive_query_prob)
        self.cross_domain_support_prob = float(cross_domain_support_prob)
        self.crop_attempts = max(1, int(crop_attempts))
        self.aug_mode = aug_mode
        self.aug_scale_limit = float(aug_scale_limit)
        self.aug_rotate_limit = int(aug_rotate_limit)
        self.train = bool(train)
        self.fixed_support = bool(fixed_support)
        if self.k_shot <= 0:
            raise ValueError("k_shot must be positive")

        query_images, query_masks = split_dirs(self.data_root, split)
        support_images, support_masks = support_dirs(self.data_root, support_root)
        self.query_map = build_path_map(query_images, query_masks)
        self.support_map = build_path_map(support_images, support_masks)
        self.query_ids = sorted(self.query_map)
        self.query_domains = {name: infer_sample_domain(name) for name in self.query_ids}
        self.support_domains = {name: infer_sample_domain(name) for name in self.support_map}
        self.positive_query_ids = self._build_positive_query_ids()
        self.support_ids = self._build_support_ids()
        if not self.support_ids:
            raise RuntimeError("No valid support samples found. Use --allow_empty_support if this is intentional.")

        self.query_transform = build_transform(
            self.img_size,
            train=train,
            scale_limit=self.aug_scale_limit,
            rotate_limit=self.aug_rotate_limit,
            aug_mode=self.aug_mode,
        )
        self.support_transform = build_transform(
            self.img_size,
            train=train,
            scale_limit=self.aug_scale_limit,
            rotate_limit=self.aug_rotate_limit,
            aug_mode=self.aug_mode,
        )

    def _build_support_ids(self):
        if self.allow_empty_support:
            return sorted(self.support_map)
        ids = [
            name
            for name in sorted(self.support_map)
            if mask_has_foreground(
                self.support_map[name]["mask"],
                min_fg_ratio=self.min_support_fg_ratio,
                max_fg_ratio=self.max_support_fg_ratio,
                mode=self.support_mask_mode,
                threshold=self.support_mask_threshold,
                foreground_value=self.support_foreground_value,
            )
        ]
        return ids

    def _build_positive_query_ids(self):
        ids = [
            name
            for name in sorted(self.query_map)
            if mask_has_foreground(
                self.query_map[name]["mask"],
                min_fg_ratio=self.min_query_fg_ratio,
                max_fg_ratio=self.max_query_fg_ratio,
                mode=self.mask_mode,
                threshold=self.mask_threshold,
                foreground_value=self.foreground_value,
            )
        ]
        return ids

    def __len__(self):
        return self.episodes_per_epoch if self.train else len(self.query_ids)

    def _load_tensor_pair(self, item, transform, support=False):
        image = load_image(item["image"])
        if support:
            mask = load_mask(
                item["mask"],
                mode=self.support_mask_mode,
                threshold=self.support_mask_threshold,
                foreground_value=self.support_foreground_value,
            )
        else:
            mask = load_mask(
                item["mask"],
                mode=self.mask_mode,
                threshold=self.mask_threshold,
                foreground_value=self.foreground_value,
            )
        if image.size != mask.size:
            mask = mask.resize(image.size, resample=Image.NEAREST)
        return apply_transform(transform, image, mask)

    def _sample_query_id(self, index):
        if self.train:
            if self.positive_query_ids and random.random() < self.positive_query_prob:
                return random.choice(self.positive_query_ids)
            return random.choice(self.query_ids)
        return self.query_ids[index % len(self.query_ids)]

    def _sample_support_ids(self, query_id, index):
        pool = [name for name in self.support_ids if name != query_id]
        if not pool:
            pool = self.support_ids
        if self.train and not self.fixed_support and self.cross_domain_support_prob > 0.0:
            query_domain = self.query_domains.get(query_id, "")
            cross_domain_pool = [
                name for name in pool if self.support_domains.get(name, "") and self.support_domains.get(name, "") != query_domain
            ]
            if cross_domain_pool and random.random() < self.cross_domain_support_prob:
                pool = cross_domain_pool
        if self.train and not self.fixed_support:
            if len(pool) >= self.k_shot:
                return random.sample(pool, self.k_shot)
            return [random.choice(pool) for _ in range(self.k_shot)]
        start = index % len(pool)
        return [pool[(start + offset) % len(pool)] for offset in range(self.k_shot)]

    @staticmethod
    def _has_foreground(mask_tensor):
        return bool((mask_tensor.float() > 0.5).any())

    def _load_query_with_retry(self, index):
        if not self.train:
            query_name = self._sample_query_id(index)
            query_image, query_mask = self._load_tensor_pair(self.query_map[query_name], self.query_transform, support=False)
            return query_name, query_image, query_mask

        last = None
        for _ in range(self.crop_attempts):
            query_name = self._sample_query_id(index)
            query_image, query_mask = self._load_tensor_pair(self.query_map[query_name], self.query_transform, support=False)
            last = (query_name, query_image, query_mask)
            if self._has_foreground(query_mask):
                return last
        return last

    def _retry_support_pool(self, query_id, used_names):
        pool = [name for name in self.support_ids if name != query_id and name not in used_names]
        if not pool:
            pool = [name for name in self.support_ids if name != query_id]
        if not pool:
            pool = list(self.support_ids)
        return pool

    def _load_support_with_retry(self, query_id, index, shot_idx, initial_name, used_names):
        support_name = initial_name
        last = None
        for attempt in range(self.crop_attempts):
            if attempt > 0:
                pool = self._retry_support_pool(query_id, used_names)
                if self.train and not self.fixed_support:
                    support_name = random.choice(pool)
                else:
                    support_name = pool[(index + attempt + shot_idx) % len(pool)]
            support_image, support_mask = self._load_tensor_pair(
                self.support_map[support_name],
                self.support_transform,
                support=True,
            )
            last = (support_name, support_image, support_mask)
            if self.allow_empty_support or self._has_foreground(support_mask):
                return last
        return last

    def __getitem__(self, index):
        query_name, query_image, query_mask = self._load_query_with_retry(index)
        support_names = self._sample_support_ids(query_name, index)
        support_images = []
        support_masks = []
        used_support_names = []
        resolved_support_names = []
        for shot_idx, name in enumerate(support_names):
            name, support_image, support_mask = self._load_support_with_retry(
                query_name,
                index,
                shot_idx,
                name,
                used_support_names,
            )
            used_support_names.append(name)
            resolved_support_names.append(name)
            support_images.append(support_image)
            support_masks.append(support_mask)

        return {
            "support_images": torch.stack(support_images, dim=0),  # [K, 3, H, W]
            "support_masks": torch.stack(support_masks, dim=0),    # [K, 1, H, W]
            "query_image": query_image,                            # [3, H, W]
            "query_mask": query_mask,                              # [1, H, W]
            "support_names": resolved_support_names,
            "query_name": query_name,
        }

