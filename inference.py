import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from datasets.crack_dataset import IMAGE_EXTS, build_path_map, load_image, load_mask
from train import build_model, load_config, merge_config, normalize_pretrained_config
from utils.checkpoint import load_checkpoint
from utils.transforms import apply_transform, build_transform
from utils.visualizer import denormalize_image


def parse_args():
    parser = argparse.ArgumentParser(description="MPDFormer inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=str(THIS_DIR / "configs" / "default.yaml"))
    parser.add_argument("--support_image", type=str, default="")
    parser.add_argument("--support_mask", type=str, default="")
    parser.add_argument("--query_image", type=str, default="")
    parser.add_argument("--support_root", type=str, default="")
    parser.add_argument("--query_root", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--backbone", type=str, choices=["b0", "b3", "b4", "b5"], default=None)
    parser.add_argument("--k_shot", type=int, default=None)
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--backbone_weights_path", "--backbone-weights-path", type=str, default=None)
    parser.add_argument("--pretrained-backbone", "--pretrained", dest="pretrained_backbone", action="store_true")
    parser.add_argument("--no-pretrained-backbone", "--no-pretrained", dest="pretrained_backbone", action="store_false")
    parser.set_defaults(pretrained_backbone=None)
    return parser.parse_args()


def image_files(root):
    root = Path(root)
    return sorted(file for file in root.iterdir() if file.is_file() and file.suffix.lower() in IMAGE_EXTS)


def load_support(args, transform):
    if args.support_root:
        root = Path(args.support_root)
        support_map = build_path_map(root / "images", root / "masks")
        names = sorted(support_map)[: int(args.k_shot)]
        if len(names) < int(args.k_shot):
            raise RuntimeError("Support folder has %d pairs, need k_shot=%d" % (len(names), int(args.k_shot)))
        images, masks = [], []
        for name in names:
            image = load_image(support_map[name]["image"])
            mask = load_mask(support_map[name]["mask"])
            image_tensor, mask_tensor = apply_transform(transform, image, mask)
            images.append(image_tensor)
            masks.append(mask_tensor)
        return torch.stack(images, 0), torch.stack(masks, 0), names

    if not args.support_image or not args.support_mask:
        raise ValueError("Provide either --support_root or both --support_image and --support_mask")
    image = load_image(Path(args.support_image))
    mask = load_mask(Path(args.support_mask))
    image_tensor, mask_tensor = apply_transform(transform, image, mask)
    return image_tensor.unsqueeze(0), mask_tensor.unsqueeze(0), [Path(args.support_image).stem]


def load_query(path, transform):
    image = load_image(Path(path))
    dummy_mask = Image.new("L", image.size, 0)
    image_tensor, _ = apply_transform(transform, image, dummy_mask)
    return image_tensor, image


def save_outputs(output_dir, name, original_image, prob, similarity_maps, threshold):
    output_dir = Path(output_dir)
    (output_dir / "prob").mkdir(parents=True, exist_ok=True)
    (output_dir / "mask").mkdir(parents=True, exist_ok=True)
    (output_dir / "overlay").mkdir(parents=True, exist_ok=True)
    (output_dir / "similarity").mkdir(parents=True, exist_ok=True)

    prob_np = prob.detach().cpu().squeeze().numpy()
    prob_img = Image.fromarray(np.clip(prob_np * 255.0, 0, 255).astype(np.uint8))
    prob_img.save(output_dir / "prob" / ("%s.png" % name))
    mask = (prob_np > threshold).astype(np.uint8) * 255
    Image.fromarray(mask).save(output_dir / "mask" / ("%s.png" % name))

    original = original_image.resize(prob_img.size, resample=Image.BILINEAR).convert("RGB")
    overlay = np.asarray(original).astype(np.float32)
    red = np.zeros_like(overlay)
    red[..., 0] = 255
    alpha = (mask > 0).astype(np.float32)[..., None] * 0.45
    overlay = overlay * (1.0 - alpha) + red * alpha
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(output_dir / "overlay" / ("%s.png" % name))

    for idx, sim in enumerate(similarity_maps):
        sim_up = F.interpolate(sim, size=prob.shape[-2:], mode="bilinear", align_corners=False)
        sim_np = sim_up.detach().cpu().squeeze().numpy()
        sim_np = (sim_np - sim_np.min()) / (sim_np.max() - sim_np.min() + 1e-6)
        Image.fromarray((sim_np * 255).astype(np.uint8)).save(output_dir / "similarity" / ("%s_s%d.png" % (name, idx + 1)))


def main():
    args = parse_args()
    raw = torch.load(args.checkpoint, map_location="cpu")
    cfg = raw.get("config", None) if isinstance(raw, dict) else None
    if not cfg:
        cfg = load_config(args.config)
    if isinstance(raw, dict) and isinstance(raw.get("hyper_parameters"), dict):
        cfg.update({key: value for key, value in raw["hyper_parameters"].items() if key in cfg})
    cfg = normalize_pretrained_config(merge_config(cfg, args))
    if args.pretrained_backbone is None:
        cfg["pretrained_backbone"] = False
    if args.k_shot is None:
        args.k_shot = int(cfg.get("k_shot", 1))
    if args.threshold is None:
        args.threshold = float(cfg.get("threshold", 0.5))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    transform = build_transform(int(cfg["img_size"]), train=False)
    support_images, support_masks, support_names = load_support(args, transform)
    support_images = support_images.unsqueeze(0).to(device).float()
    support_masks = support_masks.unsqueeze(0).to(device).float()

    queries = []
    if args.query_root:
        queries = image_files(args.query_root)
    elif args.query_image:
        queries = [Path(args.query_image)]
    else:
        raise ValueError("Provide --query_image or --query_root")

    with torch.no_grad():
        for query_path in queries:
            query_tensor, original_image = load_query(query_path, transform)
            query_tensor = query_tensor.unsqueeze(0).to(device).float()
            outputs = model(support_images, support_masks, query_tensor)
            prob = F.softmax(outputs["logits"].float(), dim=1)[:, 1:2]
            save_outputs(args.output_dir, query_path.stem, original_image, prob[0], outputs["similarity_maps_final"], args.threshold)
            print("Saved prediction for %s using support %s" % (query_path.name, ",".join(support_names)))


if __name__ == "__main__":
    main()

