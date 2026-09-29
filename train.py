import argparse
import logging
from pathlib import Path
import shutil
import sys

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import yaml

import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import EpisodeCrackDataset
from losses import MPDFormerLoss
from models import PrototypeACFSegFormer
from models.mit_encoder import PRETRAINED_URLS
from utils.metrics import BinarySegMetrics
from utils.seed import seed_everything
from utils.visualizer import save_prediction_triplet


def parse_batch_limit(value):
    text = str(value)
    if "." in text:
        return float(text)
    return int(text)


def parse_args():
    parser = argparse.ArgumentParser(description="Train MPDFormer on CrackSeg9K with Lightning")
    parser.add_argument("--config", type=str, default=str(THIS_DIR / "configs" / "default.yaml"))
    parser.add_argument("--data-root", "--data_root", dest="data_root", type=str, default=None)
    parser.add_argument("--save-dir", "--save_dir", "--default-root-dir", dest="save_dir", type=str, default=None)
    parser.add_argument("--epochs", "-e", type=int, default=None)
    parser.add_argument("--batch-size", "--batch_size", "-b", dest="batch_size", type=int, default=None)
    parser.add_argument("--learning-rate", "--lr", "-l", dest="lr", type=float, default=None)
    parser.add_argument("--weight-decay", "--weight_decay", type=float, default=None)
    parser.add_argument("--img-size", "--img_size", "--crop-size", "--patch-size", dest="img_size", type=int, default=None)
    parser.add_argument("--k-shot", "--k_shot", "--shot", dest="k_shot", type=int, default=None)
    parser.add_argument("--episodes-per-epoch", "--episodes_per_epoch", type=int, default=None)
    parser.add_argument("--support-root", "--support_root", type=str, default="")
    parser.add_argument("--train-split", "--train_split", type=str, default="train")
    parser.add_argument("--val-split", "--val_split", type=str, default="val")
    parser.add_argument("--allow-empty-support", "--allow_empty_support", dest="allow_empty_support", action="store_true", default=None)
    parser.add_argument("--no-allow-empty-support", "--no_allow_empty_support", dest="allow_empty_support", action="store_false")
    parser.add_argument("--mask-mode", "--mask_mode", choices=["bright", "dark", "nonzero", "auto"], default=None)
    parser.add_argument("--mask-threshold", "--mask_threshold", type=int, default=None)
    parser.add_argument("--foreground-value", "--foreground_value", type=int, default=None)
    parser.add_argument("--support-mask-mode", "--support_mask_mode", choices=["bright", "dark", "nonzero", "auto"], default=None)
    parser.add_argument("--support-mask-threshold", "--support_mask_threshold", type=int, default=None)
    parser.add_argument("--support-foreground-value", "--support_foreground_value", type=int, default=None)
    parser.add_argument("--min-support-fg-ratio", "--min_support_fg_ratio", type=float, default=None)
    parser.add_argument("--max-support-fg-ratio", "--max_support_fg_ratio", type=float, default=None)
    parser.add_argument("--min-query-fg-ratio", "--min_query_fg_ratio", type=float, default=None)
    parser.add_argument("--max-query-fg-ratio", "--max_query_fg_ratio", type=float, default=None)
    parser.add_argument("--positive-query-prob", "--positive_query_prob", type=float, default=None)
    parser.add_argument("--cross-domain-support-prob", "--cross_domain_support_prob", type=float, default=None)
    parser.add_argument("--crop-attempts", "--crop_attempts", type=int, default=None)
    parser.add_argument("--max-val-samples", "--max_val_samples", type=int, default=None)
    parser.add_argument("--num-workers", "--num_workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", "--prefetch_factor", type=int, default=2)
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--gradient-clipping", type=float, default=1.0)
    parser.add_argument("--load", "-f", type=str, default="", help="Load a full model/Lightning checkpoint state_dict")
    parser.add_argument("--resume", type=str, default="", help="Resume Lightning trainer from checkpoint")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--backbone", type=str, choices=["b0", "b3", "b4", "b5"], default=None)
    parser.add_argument(
        "--backbone-weights-path",
        "--backbone_weights_path",
        type=str,
        default=None,
        help="Optional local MiT backbone weights path. If empty and --pretrained-backbone is set, download from preset URL.",
    )
    pretrained_group = parser.add_mutually_exclusive_group()
    pretrained_group.add_argument(
        "--pretrained-backbone",
        "--pretrained",
        dest="pretrained_backbone",
        action="store_true",
        default=None,
        help="Load ImageNet pretrained MiT backbone",
    )
    pretrained_group.add_argument(
        "--no-pretrained-backbone",
        "--no-pretrained",
        dest="pretrained_backbone",
        action="store_false",
    )
    freeze_group = parser.add_mutually_exclusive_group()
    freeze_group.add_argument("--freeze-encoder", "--freeze_encoder", dest="freeze_encoder", action="store_true", default=None)
    freeze_group.add_argument("--no-freeze-encoder", "--no_freeze_encoder", dest="freeze_encoder", action="store_false")
    parser.add_argument("--freeze-encoder-stages", "--freeze_encoder_stages", type=int, default=None)
    parser.add_argument("--decoder-dim", "--decoder_dim", type=int, default=None)
    parser.add_argument("--decoder-dropout", "--decoder_dropout", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--ssp-mode", "--ssp_mode", type=str, choices=["soft", "topk"], default=None)
    parser.add_argument("--ssp-top-ratio", "--ssp_top_ratio", type=float, default=None)
    ssp_group = parser.add_mutually_exclusive_group()
    ssp_group.add_argument("--use-ssp", "--use_ssp", dest="use_ssp", action="store_true", default=None)
    ssp_group.add_argument("--no-use-ssp", "--no_use_ssp", dest="use_ssp", action="store_false")
    parser.add_argument("--ssp-blend", "--ssp_blend", type=float, default=None)
    parser.add_argument("--alpha-min", "--alpha_min", type=float, default=None)
    parser.add_argument("--alpha-max", "--alpha_max", type=float, default=None)
    parser.add_argument("--sep-margin", "--sep_margin", type=float, default=None)
    parser.add_argument("--aug-mode", "--aug_mode", choices=["cracknex", "generalize", "strong", "resize"], default=None)
    parser.add_argument("--aug-scale-limit", "--aug_scale_limit", type=float, default=None)
    parser.add_argument("--aug-rotate-limit", "--aug_rotate_limit", type=int, default=None)
    parser.add_argument("--warmup-epochs", "--warmup_epochs", type=int, default=None)
    parser.add_argument("--threshold", "--pred-threshold", dest="threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true", default=None)
    amp_group.add_argument("--no-amp", dest="amp", action="store_false")
    aux_group = parser.add_mutually_exclusive_group()
    aux_group.add_argument("--use-aux", "--use_aux", dest="use_aux", action="store_true", default=None)
    aux_group.add_argument("--no-use-aux", "--no_use_aux", dest="use_aux", action="store_false")
    boundary_group = parser.add_mutually_exclusive_group()
    boundary_group.add_argument("--use-boundary", "--use_boundary", dest="use_boundary", action="store_true", default=None)
    boundary_group.add_argument("--no-use-boundary", "--no_use_boundary", dest="use_boundary", action="store_false")
    parser.add_argument("--lambda-bce", "--lambda_bce", type=float, default=None)
    parser.add_argument("--lambda-dice", "--lambda_dice", type=float, default=None)
    parser.add_argument("--lambda-focal", "--lambda_focal", type=float, default=None)
    parser.add_argument("--encoder-lr-mult", "--encoder_lr_mult", type=float, default=None)
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adamw"], default=None)
    parser.add_argument("--scheduler", type=str, choices=["step", "cosine"], default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--lr-milestone-gamma", "--lr_milestone_gamma", type=float, default=None)
    bn_group = parser.add_mutually_exclusive_group()
    bn_group.add_argument("--freeze-batchnorm", "--freeze_batchnorm", dest="freeze_batchnorm", action="store_true", default=None)
    bn_group.add_argument("--no-freeze-batchnorm", "--no_freeze_batchnorm", dest="freeze_batchnorm", action="store_false")
    ema_group = parser.add_mutually_exclusive_group()
    ema_group.add_argument("--use-ema", "--use_ema", dest="use_ema", action="store_true", default=None)
    ema_group.add_argument("--no-use-ema", "--no_use_ema", dest="use_ema", action="store_false")
    parser.add_argument("--ema-decay", "--ema_decay", type=float, default=None)
    parser.add_argument("--check-val-every-n-epoch", type=int, default=1)
    parser.add_argument("--limit-train-batches", type=parse_batch_limit, default=1.0)
    parser.add_argument("--limit-val-batches", type=parse_batch_limit, default=1.0)
    parser.add_argument("--num-sanity-val-steps", type=int, default=0)
    parser.add_argument("--log-every-n-steps", type=int, default=50)
    early_group = parser.add_mutually_exclusive_group()
    early_group.add_argument("--early-stopping", "--early_stopping", dest="early_stopping", action="store_true", default=None)
    early_group.add_argument("--no-early-stopping", "--no_early_stopping", dest="early_stopping", action="store_false")
    parser.add_argument("--early-stopping-patience", "--early_stopping_patience", type=int, default=None)
    parser.add_argument("--early-stopping-min-delta", "--early_stopping_min_delta", type=float, default=None)
    return parser.parse_args()


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def merge_config(cfg, args):
    skip = {
        "config",
        "load",
        "resume",
        "support_root",
        "train_split",
        "val_split",
        "devices",
        "accelerator",
        "accumulate_grad_batches",
        "gradient_clipping",
        "prefetch_factor",
        "no_persistent_workers",
        "check_val_every_n_epoch",
        "limit_train_batches",
        "limit_val_batches",
        "num_sanity_val_steps",
        "log_every_n_steps",
    }
    for key, value in vars(args).items():
        if key in skip:
            continue
        if value is not None:
            cfg[key] = value
    return cfg


def resolve_project_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def normalize_pretrained_config(cfg):
    if "pretrained_backbone" not in cfg and "pretrained" in cfg:
        cfg["pretrained_backbone"] = cfg["pretrained"]
    cfg.pop("pretrained", None)
    return cfg


def default_backbone_weights_path(backbone):
    return THIS_DIR / "checkpoints" / ("mit_%s.pth" % backbone)


def download_backbone_weights(backbone, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.hub.download_url_to_file(PRETRAINED_URLS[backbone], str(output_path), progress=True)
    logging.info("Saved MiT-%s ImageNet weights to %s", backbone, output_path)


def resolve_backbone_weights_path(cfg):
    if not bool(cfg.get("pretrained_backbone", True)):
        return None
    raw_path = cfg.get("backbone_weights_path") or str(default_backbone_weights_path(cfg["backbone"]))
    weights_path = resolve_project_path(raw_path)
    if weights_path.exists():
        return str(weights_path)

    segformer_path = THIS_DIR / "SegFormer" / "checkpoints" / ("mit_%s.pth" % cfg["backbone"])
    if segformer_path.exists():
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(segformer_path), str(weights_path))
        logging.info("Copied MiT-%s weights from %s to %s", cfg["backbone"], segformer_path, weights_path)
        return str(weights_path)

    logging.info("MiT-%s weights not found at %s; downloading.", cfg["backbone"], weights_path)
    download_backbone_weights(cfg["backbone"], weights_path)
    return str(weights_path)


def build_model(cfg):
    return PrototypeACFSegFormer(
        backbone=cfg["backbone"],
        pretrained_backbone=bool(cfg.get("pretrained_backbone", True)),
        backbone_weights_path=cfg.get("backbone_weights_path") or None,
        decoder_dim=int(cfg.get("decoder_dim", 256)),
        temperature=float(cfg.get("temperature", 10.0)),
        freeze_encoder=bool(cfg.get("freeze_encoder", False)),
        freeze_encoder_stages=int(cfg.get("freeze_encoder_stages", 0)),
        ssp_mode=cfg.get("ssp_mode", "soft"),
        ssp_top_ratio=float(cfg.get("ssp_top_ratio", 0.15)),
        use_ssp=bool(cfg.get("use_ssp", True)),
        ssp_blend=float(cfg.get("ssp_blend", 0.5)),
        alpha_min=float(cfg.get("alpha_min", 0.05)),
        alpha_max=float(cfg.get("alpha_max", 0.95)),
        sep_margin=float(cfg.get("sep_margin", 0.1)),
        decoder_dropout=float(cfg.get("decoder_dropout", 0.2)),
        use_aux=bool(cfg.get("use_aux", True)),
        use_boundary=bool(cfg.get("use_boundary", True)),
    )


def build_loss(cfg):
    return MPDFormerLoss(
        lambda_bce=float(cfg.get("lambda_bce", 1.0)),
        lambda_dice=float(cfg.get("lambda_dice", 1.0)),
        lambda_focal=float(cfg.get("lambda_focal", 0.25)),
    )


def create_loader(cfg, split, train, support_root=None, fixed_support=False):
    data_root = resolve_project_path(cfg["data_root"])
    dataset = EpisodeCrackDataset(
        data_root=data_root,
        split=split,
        img_size=int(cfg["img_size"]),
        k_shot=int(cfg["k_shot"]),
        episodes_per_epoch=int(cfg["episodes_per_epoch"]),
        support_root=support_root,
        allow_empty_support=bool(cfg.get("allow_empty_support", False)),
        mask_mode=cfg.get("mask_mode", "bright"),
        mask_threshold=int(cfg.get("mask_threshold", 127)),
        foreground_value=int(cfg.get("foreground_value", 255)),
        support_mask_mode=cfg.get("support_mask_mode") or cfg.get("mask_mode", "bright"),
        support_mask_threshold=int(cfg.get("support_mask_threshold", cfg.get("mask_threshold", 127))),
        support_foreground_value=int(cfg.get("support_foreground_value", cfg.get("foreground_value", 255))),
        min_support_fg_ratio=float(cfg.get("min_support_fg_ratio", 0.001)),
        max_support_fg_ratio=float(cfg.get("max_support_fg_ratio", 1.0)),
        min_query_fg_ratio=float(cfg.get("min_query_fg_ratio", 0.001)),
        max_query_fg_ratio=float(cfg.get("max_query_fg_ratio", 1.0)),
        positive_query_prob=float(cfg.get("positive_query_prob", 0.7)),
        cross_domain_support_prob=float(cfg.get("cross_domain_support_prob", 0.3)),
        crop_attempts=int(cfg.get("crop_attempts", 50)),
        aug_mode=cfg.get("aug_mode", "cracknex"),
        aug_scale_limit=float(cfg.get("aug_scale_limit", 0.25)),
        aug_rotate_limit=int(cfg.get("aug_rotate_limit", 10)),
        train=train,
        fixed_support=fixed_support,
    )
    if not train and int(cfg.get("max_val_samples", 0)) > 0:
        dataset = Subset(dataset, list(range(min(int(cfg["max_val_samples"]), len(dataset)))))
    return DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]) if train else 1,
        shuffle=train,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
    )


def move_batch(batch, device):
    return {
        "support_images": batch["support_images"].to(device, non_blocking=True).float(),
        "support_masks": batch["support_masks"].to(device, non_blocking=True).float(),
        "query_image": batch["query_image"].to(device, non_blocking=True).float(),
        "query_mask": batch["query_mask"].to(device, non_blocking=True).float(),
        "query_name": batch["query_name"],
        "support_names": batch["support_names"],
    }


@torch.no_grad()
def evaluate(model, criterion, loader, device, threshold=0.5, save_dir=None, save_pred=False, amp=False):
    model.eval()
    metric = BinarySegMetrics(threshold=threshold)
    totals = {}
    steps = 0
    save_dir = Path(save_dir) if save_dir else None
    for batch in tqdm(loader, desc="val", leave=False):
        batch = move_batch(batch, device)
        with torch.cuda.amp.autocast(enabled=bool(amp and device.type == "cuda")):
            outputs = model(batch["support_images"], batch["support_masks"], batch["query_image"])
            _, loss_parts = criterion(outputs, batch["query_mask"])
        metric.update(outputs["logits"], batch["query_mask"])
        steps += 1
        for key, value in loss_parts.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        if save_pred and save_dir is not None:
            prob = F.softmax(outputs["logits"].float(), dim=1)[0, 1:2]
            name = batch["query_name"][0] if isinstance(batch["query_name"], (list, tuple)) else str(steps)
            save_prediction_triplet(save_dir / ("%s.png" % name), batch["query_image"][0], batch["query_mask"][0], prob, threshold)

    losses = {"val_" + key: value / max(1, steps) for key, value in totals.items()}
    metrics = {"val_" + key: value for key, value in metric.compute().items()}
    return {**losses, **metrics}


def load_state_for_lightning_module(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    if isinstance(checkpoint, dict) and "model" in checkpoint and "state_dict" not in checkpoint:
        state_dict = checkpoint["model"]
    cleaned = {}
    for key, value in state_dict.items():
        if key == "mask_values":
            continue
        new_key = key.replace("module.", "")
        if not new_key.startswith(("model.", "criterion.")):
            new_key = "model." + new_key
        cleaned[new_key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    return missing, unexpected


def main():
    args = parse_args()
    try:
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    except ImportError as exc:
        raise ImportError("MPDFormer Lightning training requires pytorch_lightning. Install it before training.") from exc

    from dataset import MPDFormerDataModule
    from lightning_module import LitMPDFormer

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    torch.backends.cudnn.benchmark = True
    cfg = normalize_pretrained_config(merge_config(load_config(args.config), args))
    seed_everything(int(cfg.get("seed", 42)))
    root_dir = resolve_project_path(cfg["save_dir"])
    root_dir.mkdir(parents=True, exist_ok=True)
    backbone_weights_path = resolve_backbone_weights_path(cfg)
    cfg["backbone_weights_path"] = backbone_weights_path or ""
    (root_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    data_module = MPDFormerDataModule(
        data_root=str(resolve_project_path(cfg["data_root"])),
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg.get("num_workers", 4)),
        img_size=int(cfg["img_size"]),
        k_shot=int(cfg["k_shot"]),
        episodes_per_epoch=int(cfg["episodes_per_epoch"]),
        train_split=args.train_split,
        val_split=args.val_split,
        support_root=args.support_root or None,
        allow_empty_support=bool(cfg.get("allow_empty_support", False)),
        mask_mode=cfg.get("mask_mode", "bright"),
        mask_threshold=int(cfg.get("mask_threshold", 127)),
        foreground_value=int(cfg.get("foreground_value", 255)),
        support_mask_mode=cfg.get("support_mask_mode") or cfg.get("mask_mode", "bright"),
        support_mask_threshold=int(cfg.get("support_mask_threshold", cfg.get("mask_threshold", 127))),
        support_foreground_value=int(cfg.get("support_foreground_value", cfg.get("foreground_value", 255))),
        min_support_fg_ratio=float(cfg.get("min_support_fg_ratio", 0.001)),
        max_support_fg_ratio=float(cfg.get("max_support_fg_ratio", 1.0)),
        min_query_fg_ratio=float(cfg.get("min_query_fg_ratio", 0.001)),
        max_query_fg_ratio=float(cfg.get("max_query_fg_ratio", 1.0)),
        positive_query_prob=float(cfg.get("positive_query_prob", 0.7)),
        cross_domain_support_prob=float(cfg.get("cross_domain_support_prob", 0.3)),
        crop_attempts=int(cfg.get("crop_attempts", 50)),
        aug_mode=cfg.get("aug_mode", "cracknex"),
        aug_scale_limit=float(cfg.get("aug_scale_limit", 0.25)),
        aug_rotate_limit=int(cfg.get("aug_rotate_limit", 10)),
        max_val_samples=int(cfg.get("max_val_samples", 0)),
        pin_memory=True,
        persistent_workers=not args.no_persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    data_module.setup("fit")

    logging.info(
        "Starting MPDFormer: epochs=%d batch=%d accum=%d lr=%g train=%d val=%d img=%d k_shot=%d backbone=%s pretrained_backbone=%s amp=%s",
        int(cfg["epochs"]),
        int(cfg["batch_size"]),
        args.accumulate_grad_batches,
        float(cfg["lr"]),
        len(data_module.train_set),
        len(data_module.val_set),
        int(cfg["img_size"]),
        int(cfg["k_shot"]),
        cfg["backbone"],
        bool(cfg.get("pretrained_backbone", True)),
        bool(cfg.get("amp", True)),
    )
    logging.info(
        "Mask/support settings: mask_mode=%s threshold=%d support_mask_mode=%s support_threshold=%d query_fg_ratio=[%.6f, %.3f] positive_query_prob=%.2f support_fg_ratio=[%.6f, %.3f] cross_domain_support_prob=%.2f aug_mode=%s aug_scale_limit=%.2f",
        cfg.get("mask_mode", "bright"),
        int(cfg.get("mask_threshold", 127)),
        cfg.get("support_mask_mode") or cfg.get("mask_mode", "bright"),
        int(cfg.get("support_mask_threshold", cfg.get("mask_threshold", 127))),
        float(cfg.get("min_query_fg_ratio", 0.001)),
        float(cfg.get("max_query_fg_ratio", 1.0)),
        float(cfg.get("positive_query_prob", 0.7)),
        float(cfg.get("min_support_fg_ratio", 0.001)),
        float(cfg.get("max_support_fg_ratio", 1.0)),
        float(cfg.get("cross_domain_support_prob", 0.3)),
        cfg.get("aug_mode", "cracknex"),
        float(cfg.get("aug_scale_limit", 0.25)),
    )

    model = LitMPDFormer(
        backbone=cfg["backbone"],
        pretrained_backbone=bool(cfg.get("pretrained_backbone", True)),
        backbone_weights_path=backbone_weights_path or "",
        freeze_encoder=bool(cfg.get("freeze_encoder", False)),
        freeze_encoder_stages=int(cfg.get("freeze_encoder_stages", 0)),
        decoder_dim=int(cfg.get("decoder_dim", 256)),
        decoder_dropout=float(cfg.get("decoder_dropout", 0.2)),
        temperature=float(cfg.get("temperature", 10.0)),
        ssp_mode=cfg.get("ssp_mode", "soft"),
        ssp_top_ratio=float(cfg.get("ssp_top_ratio", 0.15)),
        use_ssp=bool(cfg.get("use_ssp", True)),
        ssp_blend=float(cfg.get("ssp_blend", 0.5)),
        alpha_min=float(cfg.get("alpha_min", 0.05)),
        alpha_max=float(cfg.get("alpha_max", 0.95)),
        sep_margin=float(cfg.get("sep_margin", 0.1)),
        use_aux=bool(cfg.get("use_aux", True)),
        use_boundary=bool(cfg.get("use_boundary", True)),

        lambda_bce=float(cfg.get("lambda_bce", 1.0)),
        lambda_dice=float(cfg.get("lambda_dice", 1.0)),
        lambda_focal=float(cfg.get("lambda_focal", 0.25)),

        freeze_batchnorm=bool(cfg.get("freeze_batchnorm", False)),
        use_ema=bool(cfg.get("use_ema", False)),
        lr = cfg.get("lr", 1.0e-4),
        encoder_lr_mult = cfg.get("encoder_lr_mult", 0.5),
        optimizer = cfg["optimizer"],
        scheduler = cfg["scheduler"],
        momentum = float(cfg.get("momentum", 0.9)),
        lr_milestone_gamma = float(cfg.get("lr_milestone_gamma", 0.5)),
        max_epochs = int(cfg["epochs"]),
        warmup_epochs = int(cfg["warmup_epochs"]),
        weight_decay = float(cfg["weight_decay"]),
        pred_threshold = float(cfg.get("pred_threshold", 0.5)),
    )

    if args.load:
        missing, unexpected = load_state_for_lightning_module(model, args.load)
        logging.info("Model loaded from %s, missing=%d unexpected=%d", args.load, len(missing), len(unexpected))

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(root_dir / "checkpoints"),
        filename="checkpoint_epoch{epoch:02d}_dice{val_dice:.4f}",
        save_last=True,
        save_top_k=3,
        monitor="val_dice",
        mode="max",
    )
    miou_checkpoint = ModelCheckpoint(
        dirpath=str(root_dir / "checkpoints"),
        filename="best_miou_epoch{epoch:02d}_miou{val_miou:.4f}",
        save_top_k=1,
        monitor="val_miou",
        mode="max",
    )
    recall_checkpoint = ModelCheckpoint(
        dirpath=str(root_dir / "checkpoints"),
        filename="best_recall_epoch{epoch:02d}_recall{val_recall:.4f}",
        save_top_k=1,
        monitor="val_recall",
        mode="max",
    )
    precision = "16-mixed" if bool(cfg.get("amp", True)) else 32
    callbacks = [
        checkpoint_callback,
        miou_checkpoint,
        recall_checkpoint,
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if bool(cfg.get("early_stopping", False)):
        callbacks.append(
            EarlyStopping(
                monitor="val_dice",
                mode="max",
                patience=int(cfg.get("early_stopping_patience", 20)),
                min_delta=float(cfg.get("early_stopping_min_delta", 0.001)),
                verbose=True,
            )
        )
    trainer = pl.Trainer(
        default_root_dir=str(root_dir),
        max_epochs=int(cfg["epochs"]),
        accelerator=args.accelerator,
        devices=args.devices,
        precision=precision,
        callbacks=callbacks,
        gradient_clip_val=args.gradient_clipping,
        accumulate_grad_batches=args.accumulate_grad_batches,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        num_sanity_val_steps=args.num_sanity_val_steps,
        log_every_n_steps=args.log_every_n_steps,
        logger=True,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule=data_module, ckpt_path=args.resume or None)


if __name__ == "__main__":
    main()

