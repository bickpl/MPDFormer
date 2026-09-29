import logging
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset

from datasets import EpisodeCrackDataset


class MPDFormerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        batch_size: int = 4,
        num_workers: Optional[int] = None,
        img_size: int = 512,
        k_shot: int = 1,
        episodes_per_epoch: int = 1000,
        train_split: str = "train",
        val_split: str = "val",
        support_root: Optional[str] = None,
        allow_empty_support: bool = False,
        mask_mode: str = "bright",
        mask_threshold: int = 127,
        foreground_value: int = 255,
        support_mask_mode: Optional[str] = None,
        support_mask_threshold: Optional[int] = None,
        support_foreground_value: Optional[int] = None,
        min_support_fg_ratio: float = 0.001,
        max_support_fg_ratio: float = 1.0,
        min_query_fg_ratio: float = 0.001,
        max_query_fg_ratio: float = 1.0,
        positive_query_prob: float = 0.7,
        cross_domain_support_prob: float = 0.3,
        crop_attempts: int = 50,
        aug_mode: str = "cracknex",
        aug_scale_limit: float = 0.25,
        aug_rotate_limit: int = 10,
        max_val_samples: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = 0 if num_workers is None else num_workers
        self.img_size = img_size
        self.k_shot = k_shot
        self.episodes_per_epoch = episodes_per_epoch
        self.train_split = train_split
        self.val_split = val_split
        self.support_root = support_root
        self.allow_empty_support = allow_empty_support
        self.mask_mode = mask_mode
        self.mask_threshold = int(mask_threshold)
        self.foreground_value = int(foreground_value)
        self.support_mask_mode = support_mask_mode or mask_mode
        self.support_mask_threshold = self.mask_threshold if support_mask_threshold is None else int(support_mask_threshold)
        self.support_foreground_value = self.foreground_value if support_foreground_value is None else int(support_foreground_value)
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
        self.max_val_samples = max(0, int(max_val_samples))
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and self.num_workers > 0
        self.prefetch_factor = prefetch_factor if self.num_workers > 0 else None
        self.train_set = None
        self.val_set = None
        self.mask_values = [0, 1]

    def setup(self, stage=None):
        self.train_set = EpisodeCrackDataset(
            data_root=self.data_root,
            split=self.train_split,
            img_size=self.img_size,
            k_shot=self.k_shot,
            episodes_per_epoch=self.episodes_per_epoch,
            support_root=self.support_root,
            allow_empty_support=self.allow_empty_support,
            mask_mode=self.mask_mode,
            mask_threshold=self.mask_threshold,
            foreground_value=self.foreground_value,
            support_mask_mode=self.support_mask_mode,
            support_mask_threshold=self.support_mask_threshold,
            support_foreground_value=self.support_foreground_value,
            min_support_fg_ratio=self.min_support_fg_ratio,
            max_support_fg_ratio=self.max_support_fg_ratio,
            min_query_fg_ratio=self.min_query_fg_ratio,
            max_query_fg_ratio=self.max_query_fg_ratio,
            positive_query_prob=self.positive_query_prob,
            cross_domain_support_prob=self.cross_domain_support_prob,
            crop_attempts=self.crop_attempts,
            aug_mode=self.aug_mode,
            aug_scale_limit=self.aug_scale_limit,
            aug_rotate_limit=self.aug_rotate_limit,
            train=True,
            fixed_support=False,
        )
        val_set = EpisodeCrackDataset(
            data_root=self.data_root,
            split=self.val_split,
            img_size=self.img_size,
            k_shot=self.k_shot,
            episodes_per_epoch=self.episodes_per_epoch,
            support_root=self.support_root,
            allow_empty_support=self.allow_empty_support,
            mask_mode=self.mask_mode,
            mask_threshold=self.mask_threshold,
            foreground_value=self.foreground_value,
            support_mask_mode=self.support_mask_mode,
            support_mask_threshold=self.support_mask_threshold,
            support_foreground_value=self.support_foreground_value,
            min_support_fg_ratio=self.min_support_fg_ratio,
            max_support_fg_ratio=self.max_support_fg_ratio,
            min_query_fg_ratio=self.min_query_fg_ratio,
            max_query_fg_ratio=self.max_query_fg_ratio,
            positive_query_prob=0.0,
            cross_domain_support_prob=0.0,
            crop_attempts=self.crop_attempts,
            aug_mode=self.aug_mode,
            aug_scale_limit=self.aug_scale_limit,
            aug_rotate_limit=self.aug_rotate_limit,
            train=False,
            fixed_support=True,
        )
        if self.max_val_samples > 0:
            val_set = Subset(val_set, list(range(min(self.max_val_samples, len(val_set)))))
        self.val_set = val_set
        logging.info(
            "MPDFormer data: train_episodes=%d val_episodes=%d positive_query=%d/%d support=%d/%d mask_mode=%s mask_threshold=%d support_mask_mode=%s support_threshold=%d query_fg_ratio=[%.6f, %.3f] support_fg_ratio=[%.6f, %.3f] aug_mode=%s",
            len(self.train_set),
            len(self.val_set),
            len(self.train_set.positive_query_ids),
            len(self.train_set.query_ids),
            len(self.train_set.support_ids),
            len(self.train_set.support_map),
            self.mask_mode,
            self.mask_threshold,
            self.support_mask_mode,
            self.support_mask_threshold,
            self.min_query_fg_ratio,
            self.max_query_fg_ratio,
            self.min_support_fg_ratio,
            self.max_support_fg_ratio,
            self.aug_mode,
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

