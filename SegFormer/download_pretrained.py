import argparse
from pathlib import Path
import sys

import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from model import PRETRAINED_URLS

def main():
    parser = argparse.ArgumentParser(description="Download SegFormer MiT backbone pretrained weights")
    parser.add_argument("--variant", type=str, default="b3", choices=sorted(PRETRAINED_URLS.keys()))
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(THIS_DIR / "checkpoints"),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / ("mit_%s.pth" % args.variant)
    torch.hub.download_url_to_file(PRETRAINED_URLS[args.variant], str(output_path))
    print("Saved pretrained weights to %s" % output_path)


if __name__ == "__main__":
    main()
