"""
Dataset Setup Helper
Downloads and organizes a sample dataset for testing.
For the real project, download from Kaggle (Bone Fracture Multi-Region X-ray Data).

Usage:
  python setup_dataset.py --demo        # Creates synthetic dummy images for testing
  python setup_dataset.py --kaggle      # Instructions for Kaggle dataset
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image

def create_demo_dataset(n=30):
    """
    Creates synthetic grayscale 'X-ray-like' images for testing the pipeline.
    NOT for actual medical use — only for validating code works.
    """
    print("📁 Creating demo dataset (synthetic images)...")
    for split in ["train", "test"]:
        for label in ["fractured", "normal"]:
            path = f"dataset/{split}/{label}"
            os.makedirs(path, exist_ok=True)

            for i in range(n):
                # Simulate a bone X-ray: light gray background + dark bone shape
                img = np.ones((224, 224, 3), dtype=np.uint8) * 30

                # Bone shaft (white rod)
                img[40:190, 90:134, :] = 200

                if label == "fractured":
                    # Add a fracture line (dark diagonal)
                    for px in range(80, 150):
                        jitter = np.random.randint(-3, 3)
                        y = px
                        x = 100 + jitter
                        if 0 <= x < 224:
                            img[y, max(0,x-2):x+3, :] = 10
                    # Add noise
                    noise = np.random.randint(0, 30, (224, 224, 3), dtype=np.uint8)
                    img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
                else:
                    noise = np.random.randint(0, 15, (224, 224, 3), dtype=np.uint8)
                    img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)

                Image.fromarray(img).save(f"{path}/img_{i:03d}.png")

    print("✅ Demo dataset created:")
    print("   dataset/train/fractured/ — 30 images")
    print("   dataset/train/normal/    — 30 images")
    print("   dataset/test/fractured/  — 30 images")
    print("   dataset/test/normal/     — 30 images")
    print("\nRun: python train_model.py")


def kaggle_instructions():
    msg = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📦 REAL DATASET SETUP (Kaggle)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Option 1 — Bone Fracture Multi-Region X-ray Data
  URL : https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data
  Classes: fractured, non-fractured
  Images : ~10,000 X-ray images

  Setup steps:
    1. pip install kaggle
    2. Place your kaggle.json in ~/.kaggle/
    3. kaggle datasets download bmadushanirodrigo/fracture-multi-region-x-ray-data
    4. unzip *.zip -d dataset/
    5. python train_model.py

Option 2 — MURA (Stanford)
  URL : https://stanfordmlgroup.github.io/competitions/mura/
  Requires account registration.

Option 3 — FracAtlas
  URL : https://figshare.com/articles/dataset/The_dataset/22363012

Expected folder structure after downloading:
  dataset/
  ├── train/
  │   ├── fractured/    ← X-ray images with fractures
  │   └── normal/       ← Normal X-ray images
  └── test/
      ├── fractured/
      └── normal/
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    print(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo",   action="store_true", help="Create synthetic demo dataset")
    parser.add_argument("--kaggle", action="store_true", help="Show Kaggle dataset instructions")
    parser.add_argument("-n", type=int, default=30, help="Images per class for demo")
    args = parser.parse_args()

    if args.demo:
        create_demo_dataset(n=args.n)
    elif args.kaggle:
        kaggle_instructions()
    else:
        print("Usage:")
        print("  python setup_dataset.py --demo      # Quick synthetic test")
        print("  python setup_dataset.py --kaggle    # Real dataset instructions")