#!/usr/bin/env python3
"""
Download and organize a face recognition dataset for person identification.

Stage B classes — 5 specific persons from LFW (Labeled Faces in the Wild):
  0 - George W Bush
  1 - Colin Powell
  2 - Tony Blair
  3 - Donald Rumsfeld
  4 - Gerhard Schroeder

Stage A classes (unchanged):
  person     - LFW faces (all persons)
  no_person  - synthetic backgrounds

Usage:
    python3 download_larger_dataset.py
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path
import shutil

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
BASE_DIR    = Path("dataset")
PERSONS_DIR = BASE_DIR / "attributes"   # kept as "attributes" so preprocess works unchanged

# The 5 persons to recognize — chosen because they have the most LFW photos
TARGET_PERSONS = [
    "George_W_Bush",
    "Colin_Powell",
    "Tony_Blair",
    "Donald_Rumsfeld",
    "Gerhard_Schroeder",
]

IMG_SIZE      = 96
MAX_PER_CLASS = 400   # cap to keep dataset manageable
MIN_FACES     = 50    # minimum faces per person in LFW download

ALL_DIRS = (
    [BASE_DIR / "stage_a" / "person",
     BASE_DIR / "stage_a" / "no_person"]
    + [PERSONS_DIR / p for p in TARGET_PERSONS]
    + [BASE_DIR / "unknown_test" / f"other_person{i}" for i in range(1, 4)]
)


def save_gray(arr: np.ndarray, path: Path):
    """Save a uint8 grayscale array as PNG at IMG_SIZE x IMG_SIZE."""
    img = Image.fromarray(arr.astype(np.uint8), mode="L")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    img.save(path)


# ---------------------------------------------------------------------------
# LFW download — 5 target persons + stage_a person data + unknown_test
# ---------------------------------------------------------------------------
def download_lfw():
    from sklearn.datasets import fetch_lfw_people

    print("  Downloading LFW (min_faces_per_person=50) — first run may take a few minutes …")
    lfw = fetch_lfw_people(min_faces_per_person=MIN_FACES, resize=0.5)

    names  = lfw.target_names          # array of name strings
    images = lfw.images                # (N, H, W) float32 in [0,1]
    labels = lfw.target                # (N,) int

    print(f"  LFW loaded: {len(images)} images, {len(names)} people")

    # ---- Stage B: 5 target persons ----
    found = []
    for person_name in TARGET_PERSONS:
        # LFW names use underscores; match exactly
        matches = [i for i, n in enumerate(names) if n == person_name]
        if not matches:
            print(f"  WARNING: '{person_name}' not found in LFW — skipping")
            continue
        pid = matches[0]
        imgs = images[labels == pid]
        count = min(len(imgs), MAX_PER_CLASS)
        out_dir = PERSONS_DIR / person_name
        for j in range(count):
            arr = (imgs[j] * 255).astype(np.uint8)
            save_gray(arr, out_dir / f"{person_name.lower()}_{j+1:04d}.png")
        print(f"  {person_name}: {count} images saved")
        found.append(person_name)

    if len(found) < 5:
        print(f"\n  WARNING: Only found {len(found)}/5 target persons.")
        print("  Trying with min_faces_per_person=20 …")
        lfw2 = fetch_lfw_people(min_faces_per_person=20, resize=0.5)
        names2  = lfw2.target_names
        images2 = lfw2.images
        labels2 = lfw2.target
        for person_name in TARGET_PERSONS:
            if person_name in found:
                continue
            matches = [i for i, n in enumerate(names2) if n == person_name]
            if not matches:
                print(f"  Still not found: '{person_name}'")
                continue
            pid = matches[0]
            imgs = images2[labels2 == pid]
            count = min(len(imgs), MAX_PER_CLASS)
            out_dir = PERSONS_DIR / person_name
            for j in range(count):
                arr = (imgs[j] * 255).astype(np.uint8)
                save_gray(arr, out_dir / f"{person_name.lower()}_{j+1:04d}.png")
            print(f"  {person_name}: {count} images saved (from relaxed fetch)")
            found.append(person_name)

    # ---- Stage A: all persons ----
    stage_a_dir = BASE_DIR / "stage_a" / "person"
    count = 0
    for i, img in enumerate(images):
        if count >= MAX_PER_CLASS:
            break
        arr = (img * 255).astype(np.uint8)
        save_gray(arr, stage_a_dir / f"person_{i+1:04d}.png")
        count += 1
    print(f"  Stage A person: {count} images saved")

    # ---- Unknown test: persons NOT in our target list ----
    other_ids = [i for i, n in enumerate(names) if n not in TARGET_PERSONS]
    slot = 0
    for pid in other_ids[:3]:
        slot += 1
        ud = BASE_DIR / "unknown_test" / f"other_person{slot}"
        imgs = images[labels == pid]
        for j, img in enumerate(imgs[:50]):
            arr = (img * 255).astype(np.uint8)
            save_gray(arr, ud / f"unknown_{pid}_{j+1:03d}.png")
        print(f"  Unknown {slot} ({names[pid]}): {min(len(imgs),50)} images")

    return found


# ---------------------------------------------------------------------------
# Synthetic backgrounds (stage_a / no_person)
# ---------------------------------------------------------------------------
def generate_backgrounds(n: int = 200):
    bg_dir = BASE_DIR / "stage_a" / "no_person"
    rng = np.random.default_rng(seed=1)
    for i in range(n):
        mode = i % 4
        if mode == 0:
            img = rng.integers(0, 256, (IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        elif mode == 1:
            row = np.linspace(0, 255, IMG_SIZE).astype(np.uint8)
            img = np.tile(row, (IMG_SIZE, 1)) if i % 2 == 0 else np.tile(row.reshape(-1, 1), (1, IMG_SIZE))
        elif mode == 2:
            gray = int(rng.integers(50, 200))
            noise = rng.integers(-30, 30, (IMG_SIZE, IMG_SIZE))
            img = np.clip(gray + noise, 0, 255).astype(np.uint8)
        else:
            img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            block = 12
            for r in range(0, IMG_SIZE, block):
                for c in range(0, IMG_SIZE, block):
                    val = int(rng.integers(150, 255)) if (r // block + c // block) % 2 == 0 else int(rng.integers(0, 100))
                    img[r:r+block, c:c+block] = val
        Image.fromarray(img, mode="L").save(bg_dir / f"bg_{i+1:03d}.png")
    print(f"  Generated {n} background images → stage_a/no_person/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 55)
    print("Face Recognition Dataset Setup — 5 Person Recognizer")
    print("=" * 55)
    print(f"\nTarget persons: {TARGET_PERSONS}\n")

    # Clean and recreate directory structure
    if BASE_DIR.exists():
        shutil.rmtree(BASE_DIR)
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)

    # Download LFW
    found = download_lfw()

    # Synthetic backgrounds
    generate_backgrounds()

    # Summary
    print("\n" + "=" * 55)
    print("Dataset Ready!")
    print("=" * 55)
    print(f"\nPersons recognized ({len(found)}/5):")
    for name in TARGET_PERSONS:
        d = PERSONS_DIR / name
        n = len(list(d.glob("*.png"))) if d.exists() else 0
        status = "OK" if n > 0 else "MISSING"
        print(f"  [{status}] {name.replace('_', ' ')}: {n} images")

    print("\nStage A:")
    for sub in ["person", "no_person"]:
        d = BASE_DIR / "stage_a" / sub
        n = len(list(d.glob("*.png"))) if d.exists() else 0
        print(f"  {sub:12s}: {n} images")

    print("\nNext steps:")
    print("  python3 C_preprocess_and_augment.py --dataset_dir dataset --output_dir processed --augment_train --augmentations 5")
    print("  python3 E_train_model.py --data_dir processed --output_dir models --train_stage b")
    print("  python3 F_quantize_model.py --model_dir models --data_dir processed")


if __name__ == "__main__":
    main()
