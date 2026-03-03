#!/usr/bin/env python3
"""
Download and generate Stage A dataset for face detection.

Stage A classes:
  person     — LFW face images (500 images)
  no_person  — synthetic backgrounds (500 images)

Usage:
    python3 download_larger_dataset.py
"""

import numpy as np
from PIL import Image
from pathlib import Path
import shutil

BASE_DIR     = Path("dataset")
STAGE_A_DIR  = BASE_DIR / "stage_a"
PERSON_DIR   = STAGE_A_DIR / "person"
NO_PERSON_DIR = STAGE_A_DIR / "no_person"

IMG_SIZE     = 96
MAX_PERSONS  = 500   # person images
MAX_BG       = 500   # no_person images


def save_gray(arr: np.ndarray, path: Path):
    img = Image.fromarray(arr.astype(np.uint8), mode="L")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    img.save(path)


def download_lfw_persons():
    from sklearn.datasets import fetch_lfw_people

    print("  Downloading LFW faces (min_faces_per_person=20) ...")
    lfw = fetch_lfw_people(min_faces_per_person=20, resize=0.5)
    images = lfw.images   # (N, H, W) float32
    print(f"  LFW loaded: {len(images)} images total")

    count = min(len(images), MAX_PERSONS)
    for i in range(count):
        arr = (images[i] * 255).astype(np.uint8)
        save_gray(arr, PERSON_DIR / f"person_{i+1:04d}.png")

    print(f"  Saved {count} person images → stage_a/person/")
    return count


def generate_backgrounds():
    """Generate varied synthetic backgrounds — no faces."""
    rng = np.random.default_rng(seed=42)
    n = MAX_BG
    patterns = 6  # more variety than before

    for i in range(n):
        mode = i % patterns

        if mode == 0:
            # Pure random noise
            img = rng.integers(0, 256, (IMG_SIZE, IMG_SIZE), dtype=np.uint8)

        elif mode == 1:
            # Horizontal gradient
            row = np.linspace(0, 255, IMG_SIZE).astype(np.uint8)
            img = np.tile(row, (IMG_SIZE, 1))

        elif mode == 2:
            # Vertical gradient
            col = np.linspace(0, 255, IMG_SIZE).astype(np.uint8)
            img = np.tile(col.reshape(-1, 1), (1, IMG_SIZE))

        elif mode == 3:
            # Uniform gray + noise
            gray = int(rng.integers(40, 220))
            noise = rng.integers(-40, 40, (IMG_SIZE, IMG_SIZE))
            img = np.clip(gray + noise, 0, 255).astype(np.uint8)

        elif mode == 4:
            # Checkerboard
            img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            block = rng.integers(8, 24)
            for r in range(0, IMG_SIZE, block):
                for c in range(0, IMG_SIZE, block):
                    val = int(rng.integers(150, 255)) if (r // block + c // block) % 2 == 0 else int(rng.integers(0, 80))
                    img[r:r+block, c:c+block] = val

        else:
            # Diagonal stripes
            img = np.fromfunction(
                lambda r, c: ((r + c) * 255 // (IMG_SIZE * 2)).astype(np.uint8),
                (IMG_SIZE, IMG_SIZE), dtype=np.float32
            ).astype(np.uint8)

        Image.fromarray(img, mode="L").save(NO_PERSON_DIR / f"bg_{i+1:04d}.png")

    print(f"  Generated {n} background images → stage_a/no_person/")
    return n


def main():
    print("=" * 55)
    print("  Stage A Dataset Setup — Face Detection Only")
    print("=" * 55)

    if BASE_DIR.exists():
        shutil.rmtree(BASE_DIR)
    PERSON_DIR.mkdir(parents=True, exist_ok=True)
    NO_PERSON_DIR.mkdir(parents=True, exist_ok=True)

    persons  = download_lfw_persons()
    bg_count = generate_backgrounds()

    print("\n" + "=" * 55)
    print("Dataset Ready!")
    print("=" * 55)
    print(f"  person:    {persons} images")
    print(f"  no_person: {bg_count} images")
    print(f"  Total:     {persons + bg_count} images (balanced)")
    print("\nNext steps:")
    print("  python3 C_preprocess_and_augment.py --dataset_dir dataset --output_dir processed --augment_train --augmentations 5")
    print("  python3 E_train_model.py --data_dir processed --output_dir models")
    print("  python3 F_quantize_model.py --model_dir models --data_dir processed")


if __name__ == "__main__":
    main()
