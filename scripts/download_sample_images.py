"""
Download the first N instances of the Robo2VLM-1 dataset and save their images as JPEG.

Usage:
    python scripts/download_sample_images.py                   # first 10 entries
    python scripts/download_sample_images.py --n 20 --out data/samples
"""

import argparse
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Number of instances to download")
    parser.add_argument("--out", default="data/sample_images", help="Output directory")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("keplerccc/Robo2VLM-1", split="train", streaming=True)

    for i, entry in enumerate(ds):
        if i >= args.n:
            break
        entry_id = entry["id"]
        img = entry["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        path = out / f"{i:03d}_{entry_id}.jpg"
        img.save(path, format="JPEG")
        print(f"[{i+1}/{args.n}] saved {path}")

    print(f"\nDone. {args.n} images in {out}/")


if __name__ == "__main__":
    main()
