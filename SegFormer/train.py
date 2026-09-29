import argparse
import logging
from pathlib import Path
import sys

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from SegFormer.dataset import Crack500DataModule
from SegFormer.lightning_module import LitSegFormer
from SegFormer.model import PRETRAINED_URLS


def parse_batch_limit(value):
    text = str(value)
    if "." in text:
        return float(text)
    return int(text)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SegFormer on CRACK500 with Lightning")
    parser.add_argument("--data-root", type=str, default=str(THIS_DIR / "data" / "CRACK500"))
    parser.add_argument("--epochs", "-e", type=int, default=500, help="Number of epochs")
    parser.add_argument("--batch-size", "-b", dest="batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", "-l", dest="lr", type=float, default=6e-5, help="Backbone learning rate")
    parser.add_argument("--head-lr-mult", type=float, default=10.0, help="Decode head LR multiplier")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-iters", type=int, default=1500)
    parser.add_argument("--poly-power", type=float, default=1.0)
    parser.add_argument("--scale", "-s", type=float, default=1.0, help="Base image scale before random ratio")
    parser.add_argument("--crop-size", "--patch-size", dest="crop_size", type=int, default=256, help="Random crop size for SegFormer-style training")
    parser.add_argument("--ratio-min", type=float, default=0.5, help="Minimum random resize ratio")
    parser.add_argument("--ratio-max", type=float, default=2.0, help="Maximum random resize ratio")
    parser.add_argument("--repeat", type=int, default=50, help="Repeat training images per epoch for random resize/crop sampling")
    parser.add_argument("--amp", action="store_true", default=False, help="Use mixed precision")
    parser.add_argument("--classes", "-c", type=int, default=1, help="Number of output logits; crack segmentation uses 1 for BCE + Dice")
    parser.add_argument("--channels", type=int, default=3, help="Number of input image channels")
    parser.add_argument("--num-workers", type=int, default=16, help="Dataloader workers")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="Dataloader prefetch factor when num_workers > 0")
    parser.add_argument("--no-persistent-workers", action="store_true", help="Disable persistent dataloader workers")
    parser.add_argument("--gradient-clipping", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--load", "-f", type=str, default="", help="Load a full Lightning/model checkpoint, not MiT backbone weights")
    parser.add_argument("--default-root-dir", type=str, default=str(THIS_DIR / "runs" / "SegFormerb0ce_dice"))
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--variant", type=str, default="b0", choices=["b0", "b1", "b2", "b3", "b4", "b5"])
    pretrained_group = parser.add_mutually_exclusive_group()
    pretrained_group.add_argument("--pretrained-backbone", dest="pretrained_backbone", action="store_true", default=True, help="Load ImageNet pretrained MiT backbone")
    pretrained_group.add_argument("--no-pretrained-backbone", dest="pretrained_backbone", action="store_false")
    parser.add_argument(
        "--backbone-weights-path",
        type=str,
        default=str(THIS_DIR / "checkpoints" / "mit_b0.pth"),
        help="Optional local backbone weights path. If empty and --pretrained-backbone is set, download from preset URL.",
    )
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--check-val-every-n-epoch", type=int, default=5)
    parser.add_argument("--limit-train-batches", type=parse_batch_limit, default=1.0)
    parser.add_argument("--limit-val-batches", type=parse_batch_limit, default=1.0)
    return parser.parse_args()


def default_backbone_weights_path(variant):
    return THIS_DIR / "checkpoints" / ("mit_%s.pth" % variant)


def download_backbone_weights(variant, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.hub.download_url_to_file(PRETRAINED_URLS[variant], str(output_path), progress=True)
    logging.info("Saved MiT-%s ImageNet weights to %s", variant, output_path)


def resolve_backbone_weights_path(args):
    if not args.pretrained_backbone:
        return None
    weights_path = Path(args.backbone_weights_path) if args.backbone_weights_path else default_backbone_weights_path(args.variant)
    if weights_path.exists():
        return str(weights_path)

    logging.info("MiT-%s weights not found at %s; downloading.", args.variant, weights_path)
    download_backbone_weights(args.variant, weights_path)
    return str(weights_path)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    torch.backends.cudnn.benchmark = True
    backbone_weights_path = resolve_backbone_weights_path(args)

    data_module = Crack500DataModule(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_scale=args.scale,
        crop_size=args.crop_size,
        ratio_min=args.ratio_min,
        ratio_max=args.ratio_max,
        repeat=args.repeat,
        pin_memory=True,
        persistent_workers=not args.no_persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    data_module.setup("fit")

    train_size = len(data_module.train_set)
    val_size = len(data_module.val_set)
    logging.info(
        "Starting training: epochs=%d batch_size=%d lr=%g head_lr_mult=%g train=%d val=%d img_scale=%.3f crop_size=%d ratio=(%.2f, %.2f) repeat=%d variant=%s pretrained_backbone=%s amp=%s",
        args.epochs,
        args.batch_size,
        args.lr,
        args.head_lr_mult,
        train_size,
        val_size,
        args.scale,
        args.crop_size,
        args.ratio_min,
        args.ratio_max,
        args.repeat,
        args.variant,
        args.pretrained_backbone,
        args.amp,
    )

    model = LitSegFormer(
        n_channels=args.channels,
        n_classes=args.classes,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        head_lr_mult=args.head_lr_mult,
        warmup_iters=args.warmup_iters,
        poly_power=args.poly_power,
        amp=args.amp,
        variant=args.variant,
        pretrained_backbone=args.pretrained_backbone,
        backbone_weights_path=backbone_weights_path,
    )

    if args.load:
        checkpoint = torch.load(args.load, map_location="cpu")
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        if "mask_values" in state_dict:
            del state_dict["mask_values"]
        model.load_state_dict(state_dict, strict=False)
        logging.info("Model loaded from %s", args.load)

    checkpoint_dir = Path(args.default_root_dir) / "checkpoints"
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="checkpoint_epoch{epoch:02d}_dice{val_dice:.4f}",
        save_last=True,
        save_top_k=3,
        monitor="val_dice",
        mode="max",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    precision = "16-mixed" if args.amp else 32

    trainer = pl.Trainer(
        default_root_dir=args.default_root_dir,
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=precision,
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=args.gradient_clipping,
        accumulate_grad_batches=args.accumulate_grad_batches,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        log_every_n_steps=50,
        logger=True
    )
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
