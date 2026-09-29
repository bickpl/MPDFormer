import random
from pathlib import Path

import numpy as np
from PIL import Image
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dice import dice_coeff, dice_loss
from .model import SegFormer



class LitSegFormer(pl.LightningModule):
    def __init__(
        self,
        n_channels=3,
        n_classes=1,
        learning_rate=6e-5,
        weight_decay=0.01,
        head_lr_mult=10.0,
        warmup_iters=1500,
        min_lr_ratio=0.0,
        poly_power=1.0,
        amp=False,
        variant="b0",
        pretrained_backbone=True,
        backbone_weights_path=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = SegFormer(
            n_channels=n_channels,
            n_classes=n_classes,
            variant=variant,
            pretrained_backbone=pretrained_backbone,
            backbone_weights_path=backbone_weights_path,
        )
        self.model = self.model.to(memory_format=torch.channels_last)
        if n_classes != 1:
            raise ValueError("Crack segmentation is configured as binary segmentation; use n_classes=1 for BCE + Dice.")
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.mask_values = None
        self.example_input_array = torch.randn(1, n_channels, 256, 256)
        self._val_vis_sample = None
        self._val_vis_seen = 0

    @property
    def n_channels(self):
        return self.model.n_channels

    @property
    def n_classes(self):
        return self.model.n_classes

    def forward(self, x):
        return self.model(x)

    def setup(self, stage=None):
        datamodule = self.trainer.datamodule if self.trainer else None
        if datamodule is not None and getattr(datamodule, "mask_values", None) is not None:
            self.mask_values = datamodule.mask_values

    def _shared_loss(self, images, true_masks):
        images = images.to(dtype=torch.float32, memory_format=torch.channels_last)
        masks_pred = self(images)
        logits = masks_pred.squeeze(1)
        valid = true_masks != 255
        target = (true_masks == 1).float()
        bce_map = self.criterion(logits, target)
        bce = (bce_map * valid.float()).sum() / valid.float().sum().clamp_min(1.0)
        prob = torch.sigmoid(logits) * valid.float()
        target = target * valid.float()
        loss = bce + dice_loss(prob, target, multiclass=False)
        return masks_pred, loss

    def training_step(self, batch, batch_idx):
        images = batch["image"]
        true_masks = batch["mask"]
        assert images.shape[1] == self.n_channels, (
            "Network expects %d channels but got %d" % (self.n_channels, images.shape[1])
        )
        _, loss = self._shared_loss(images, true_masks)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=images.shape[0])
        return loss

    def on_validation_epoch_start(self):
        if self.trainer is not None and self.trainer.sanity_checking:
            return
        self._val_vis_sample = None
        self._val_vis_seen = 0

    @staticmethod
    def _tensor_image_to_uint8(image_tensor):
        image = image_tensor.detach().cpu().float()
        if image.shape[0] == 3:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = image * std + mean
        image = image.numpy()
        if image.ndim == 3:
            image = np.transpose(image, (1, 2, 0))
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
        if image.ndim == 3 and image.shape[2] == 1:
            image = image[:, :, 0]
        return image

    @staticmethod
    def _mask_to_uint8(mask_tensor):
        mask = mask_tensor.detach().cpu().numpy().astype(np.uint8)
        if mask.max() > 0:
            mask = (mask > 0).astype(np.uint8) * 255
        return mask

    def _prediction_to_uint8(self, pred_tensor):
        pred = (torch.sigmoid(pred_tensor) > 0.5).float().squeeze(0)
        return self._mask_to_uint8(pred)

    def _maybe_store_random_val_sample(self, images, true_masks, masks_pred):
        batch_size = images.shape[0]
        for i in range(batch_size):
            self._val_vis_seen += 1
            if random.randint(1, self._val_vis_seen) != 1:
                continue
            self._val_vis_sample = {
                "image": images[i].detach().cpu(),
                "mask": true_masks[i].detach().cpu(),
                "pred": masks_pred[i].detach().cpu(),
            }

    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        true_masks = batch["mask"]
        masks_pred, loss = self._shared_loss(images, true_masks)
        self._maybe_store_random_val_sample(images, true_masks, masks_pred)

        valid = true_masks != 255
        pred_mask = (torch.sigmoid(masks_pred.squeeze(1)) > 0.5).float() * valid.float()
        target = (true_masks == 1).float() * valid.float()
        dice = dice_coeff(pred_mask, target, reduce_batch_first=False)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=images.shape[0])
        self.log("val_dice", dice, on_step=False, on_epoch=True, prog_bar=True, batch_size=images.shape[0])
        return {"val_loss": loss, "val_dice": dice}

    def on_validation_epoch_end(self):
        if self._val_vis_sample is None or self.trainer is None or self.trainer.sanity_checking:
            return
        image = self._tensor_image_to_uint8(self._val_vis_sample["image"])
        mask = self._mask_to_uint8(self._val_vis_sample["mask"])
        pred = self._prediction_to_uint8(self._val_vis_sample["pred"])
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        mask_rgb = np.stack([mask] * 3, axis=-1)
        pred_rgb = np.stack([pred] * 3, axis=-1)
        combined = np.concatenate([image, mask_rgb, pred_rgb], axis=1)
        save_dir = Path(self.trainer.default_root_dir) / "val_samples"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / ("epoch_%04d.png" % (self.current_epoch + 1))
        Image.fromarray(combined).save(save_path)

    def configure_optimizers(self):
        backbone_params = []
        head_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("model.backbone"):
                backbone_params.append(param)
            else:
                head_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.hparams.learning_rate},
                {"params": head_params, "lr": self.hparams.learning_rate * self.hparams.head_lr_mult},
            ],
            weight_decay=self.hparams.weight_decay,
        )

        try:
            total_steps = int(self.trainer.estimated_stepping_batches)
        except Exception:
            total_steps = 1
        total_steps = max(1, total_steps)
        warmup_iters = min(int(self.hparams.warmup_iters), total_steps)

        def poly_warmup_lambda(step):
            if warmup_iters > 0 and step < warmup_iters:
                return float(step + 1) / float(warmup_iters)
            progress = float(step - warmup_iters) / float(max(1, total_steps - warmup_iters))
            factor = (1.0 - progress) ** float(self.hparams.poly_power)
            return max(float(self.hparams.min_lr_ratio), factor)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_warmup_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def on_save_checkpoint(self, checkpoint):
        if self.mask_values is not None:
            checkpoint["mask_values"] = self.mask_values
