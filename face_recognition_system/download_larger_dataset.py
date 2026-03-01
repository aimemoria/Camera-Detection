#!/usr/bin/env python3
"""
Download and organize a face attribute dataset for detection feedback.

Attribute classes (Stage B):
  0 - happy      : smiling / happy faces  (FER2013 happy)
  1 - sad        : sad faces              (FER2013 sad)
  2 - grief      : angry / grief faces   (FER2013 angry + disgust)
  3 - hair       : person with hair       (LFW neutral faces)
  4 - no_hair    : bald person            (CelebA Bald=True subset)
  5 - multiple   : multiple persons       (composite of 2 faces)

Stage A classes (unchanged):
  person     - LFW faces
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
BASE_DIR = Path("dataset")
ATTR_DIR = BASE_DIR / "attributes"

ATTR_CLASSES = ["happy", "sad", "grief", "hair", "no_hair", "multiple"]

ALL_DIRS = (
    [BASE_DIR / "stage_a" / "person",
     BASE_DIR / "stage_a" / "no_person"]
    + [ATTR_DIR / c for c in ATTR_CLASSES]
    + [BASE_DIR / "unknown_test" / f"other_person{i}" for i in range(1, 4)]
)

IMG_SIZE = 96
MAX_PER_CLASS = 400   # cap to keep dataset manageable
MAX_UNKNOWN   = 50


def save_gray(arr: np.ndarray, path: Path):
    """Save a uint8 grayscale array as PNG."""
    img = Image.fromarray(arr.astype(np.uint8), mode="L")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    img.save(path)


# ---------------------------------------------------------------------------
# FER2013  (happy / sad / grief)
# ---------------------------------------------------------------------------
def download_fer2013():
    """
    Download FER2013 via tensorflow_datasets.
    Returns dict: {label_int -> list of (96,96) uint8 arrays}
    FER2013 emotions: 0=angry 1=disgust 2=fear 3=happy 4=sad 5=surprise 6=neutral
    """
    import tensorflow_datasets as tfds

    print("  Downloading FER2013 via tensorflow_datasets …")
    ds, info = tfds.load("fer2013", split="train", with_info=True, as_supervised=False)

    buckets = {k: [] for k in range(7)}
    for sample in ds.as_numpy_iterator():
        label = int(sample["label"])
        img   = sample["image"]          # (48, 48, 1) uint8
        img2d = img[:, :, 0]             # grayscale 2-D
        buckets[label].append(img2d)

    print(f"  FER2013 loaded: { {k: len(v) for k, v in buckets.items()} }")
    return buckets


def save_fer2013_classes(buckets, attr_dir: Path):
    """
    Map FER2013 emotion buckets → attribute folders.
      happy  ← emotion 3
      sad    ← emotion 4
      grief  ← emotions 0 (angry) + 1 (disgust)
    """
    mapping = {
        "happy": buckets[3],
        "sad":   buckets[4],
        "grief": buckets[0] + buckets[1],
    }

    for cls_name, images in mapping.items():
        out_dir = attr_dir / cls_name
        count = min(len(images), MAX_PER_CLASS)
        indices = np.random.choice(len(images), count, replace=False)
        for j, idx in enumerate(indices):
            save_gray(images[idx], out_dir / f"{cls_name}_{j+1:04d}.png")
        print(f"  Saved {count} images → attributes/{cls_name}/")

    # Return neutral (6) for hair class
    return buckets[6]


# ---------------------------------------------------------------------------
# LFW  (hair class + stage_a person + unknown_test)
# ---------------------------------------------------------------------------
def download_lfw(base_dir: Path, attr_dir: Path):
    """
    Download LFW faces.
    - Most LFW subjects have visible hair → use as 'hair' attribute class
    - Also populates stage_a/person and unknown_test/
    """
    from sklearn.datasets import fetch_lfw_people

    print("  Downloading LFW (min_faces_per_person=30) …")
    lfw = fetch_lfw_people(min_faces_per_person=30, resize=0.5)

    unique, counts = np.unique(lfw.target, return_counts=True)
    sorted_idx = np.argsort(counts)[::-1]

    # First 5 people → stage_a/person
    # Next 3 people  → unknown_test
    # All the rest   → hair attribute class
    stage_a_ids  = sorted_idx[:5]
    unknown_ids  = sorted_idx[5:8]
    hair_ids     = sorted_idx[8:]

    # ---- stage_a / person ----
    stage_a_person_dir = base_dir / "stage_a" / "person"
    for pid in stage_a_ids:
        imgs = lfw.images[lfw.target == pid]
        for j, img in enumerate(imgs):
            arr = (img * 255 / img.max()).astype(np.uint8)
            save_gray(arr, stage_a_person_dir / f"person_{pid}_{j+1:03d}.png")

    total_a = len(list(stage_a_person_dir.glob("*.png")))
    print(f"  Stage A person images: {total_a}")

    # ---- unknown_test ----
    for i, pid in enumerate(unknown_ids):
        ud = base_dir / "unknown_test" / f"other_person{i+1}"
        imgs = lfw.images[lfw.target == pid]
        for j, img in enumerate(imgs[:MAX_UNKNOWN]):
            arr = (img * 255 / img.max()).astype(np.uint8)
            save_gray(arr, ud / f"unknown_{pid}_{j+1:03d}.png")
        print(f"  Unknown {i+1}: {min(len(imgs), MAX_UNKNOWN)} images")

    # ---- hair attribute ----
    hair_dir = attr_dir / "hair"
    count = 0
    for pid in hair_ids:
        if count >= MAX_PER_CLASS:
            break
        imgs = lfw.images[lfw.target == pid]
        for j, img in enumerate(imgs):
            if count >= MAX_PER_CLASS:
                break
            arr = (img * 255 / img.max()).astype(np.uint8)
            save_gray(arr, hair_dir / f"hair_{pid}_{j+1:03d}.png")
            count += 1

    print(f"  Saved {count} images → attributes/hair/")
    return lfw  # return for compositing multiple faces


# ---------------------------------------------------------------------------
# CelebA bald subset  (no_hair class)
# ---------------------------------------------------------------------------
def download_celeba_bald(attr_dir: Path):
    """
    Download CelebA and save images where Bald attribute == True.
    Uses tensorflow_datasets. This requires ~2GB download on first run.
    """
    import tensorflow_datasets as tfds

    print("  Downloading CelebA bald subset via tensorflow_datasets …")
    print("  (First run downloads ~2 GB — this may take several minutes)")

    ds = tfds.load("celeb_a", split="train", as_supervised=False)
    no_hair_dir = attr_dir / "no_hair"
    count = 0

    for sample in ds.as_numpy_iterator():
        if count >= MAX_PER_CLASS:
            break
        if not sample["attributes"]["Bald"]:
            continue
        img = sample["image"]           # (218, 178, 3) uint8
        gray = np.mean(img, axis=2).astype(np.uint8)   # convert to grayscale
        save_gray(gray, no_hair_dir / f"no_hair_{count+1:04d}.png")
        count += 1
        if count % 50 == 0:
            print(f"    {count} bald images saved …")

    print(f"  Saved {count} images → attributes/no_hair/")
    return count


# ---------------------------------------------------------------------------
# Multiple-persons composites
# ---------------------------------------------------------------------------
def generate_multiple(lfw, attr_dir: Path):
    """
    Create 'multiple persons' images by tiling two LFW faces side-by-side
    in a 96x96 canvas (each face scaled to 48x96 then placed left/right).
    """
    print("  Generating multiple-persons composites …")
    multi_dir = attr_dir / "multiple"
    half = IMG_SIZE // 2
    rng  = np.random.default_rng(seed=0)
    count = 0

    indices = rng.permutation(len(lfw.images))
    i = 0
    while count < MAX_PER_CLASS and i + 1 < len(indices):
        img_left  = lfw.images[indices[i]]
        img_right = lfw.images[indices[i + 1]]

        # Convert to uint8
        left_u8  = (img_left  * 255 / img_left.max()).astype(np.uint8)
        right_u8 = (img_right * 255 / img_right.max()).astype(np.uint8)

        canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

        # Resize each face to (IMG_SIZE, half) then place side by side
        left_pil  = Image.fromarray(left_u8,  mode="L").resize((half, IMG_SIZE), Image.Resampling.LANCZOS)
        right_pil = Image.fromarray(right_u8, mode="L").resize((half, IMG_SIZE), Image.Resampling.LANCZOS)

        canvas[:, :half]  = np.array(left_pil)
        canvas[:, half:]  = np.array(right_pil)

        Image.fromarray(canvas, mode="L").save(
            multi_dir / f"multiple_{count+1:04d}.png"
        )
        count += 1
        i += 2

    print(f"  Saved {count} images → attributes/multiple/")


# ---------------------------------------------------------------------------
# Synthetic backgrounds (stage_a / no_person)
# ---------------------------------------------------------------------------
def generate_backgrounds(base_dir: Path, n: int = 200):
    bg_dir = base_dir / "stage_a" / "no_person"
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
    print("Face Attribute Detection Dataset Setup")
    print("=" * 55)
    print(f"\nAttribute classes: {ATTR_CLASSES}\n")

    # Clean and recreate directory structure
    if BASE_DIR.exists():
        shutil.rmtree(BASE_DIR)
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)

    # ---- FER2013 (happy / sad / grief) ----
    neutral_images = []
    fer_ok = False
    try:
        import tensorflow_datasets as tfds  # noqa: F401
        buckets = download_fer2013()
        neutral_images = buckets[6]  # neutral faces for hair fallback
        save_fer2013_classes(buckets, ATTR_DIR)
        fer_ok = True
    except Exception as e:
        print(f"  FER2013 unavailable ({e}). Using Olivetti fallback for emotions …")
        # Olivetti fallback for happy/sad/grief
        from sklearn.datasets import fetch_olivetti_faces
        data = fetch_olivetti_faces()
        imgs, labels = data.images, data.target
        # Use different Olivetti persons as proxies for emotion classes
        for cls_name, person_ids in [("happy", [0, 1, 2, 3]),
                                      ("sad",   [4, 5, 6, 7]),
                                      ("grief", [8, 9, 10, 11])]:
            out_dir = ATTR_DIR / cls_name
            count = 0
            for pid in person_ids:
                for j, img in enumerate(imgs[labels == pid]):
                    arr = (img * 255).astype(np.uint8)
                    save_gray(arr, out_dir / f"{cls_name}_{count+1:04d}.png")
                    count += 1
            print(f"  Saved {count} images → attributes/{cls_name}/ (Olivetti proxy)")

    # ---- LFW (hair + stage_a/person + unknown_test) ----
    lfw = None
    try:
        from sklearn.datasets import fetch_lfw_people  # noqa: F401
        lfw = download_lfw(BASE_DIR, ATTR_DIR)
    except Exception as e:
        print(f"  LFW unavailable ({e}). Using Olivetti for hair/stage_a …")
        from sklearn.datasets import fetch_olivetti_faces
        data = fetch_olivetti_faces()
        imgs, labels = data.images, data.target
        hair_dir = ATTR_DIR / "hair"
        count = 0
        for pid in range(20, 35):
            for j, img in enumerate(imgs[labels == pid]):
                arr = (img * 255).astype(np.uint8)
                save_gray(arr, hair_dir / f"hair_{count+1:04d}.png")
                count += 1
                # Also stage_a
                save_gray(arr, BASE_DIR / "stage_a" / "person" / f"sa_{count:04d}.png")
        print(f"  Saved {count} images → attributes/hair/ (Olivetti proxy)")
        lfw = None

    # ---- CelebA bald (no_hair) ----
    celeba_ok = False
    try:
        import tensorflow_datasets as tfds  # noqa: F401
        n_bald = download_celeba_bald(ATTR_DIR)
        if n_bald > 0:
            celeba_ok = True
    except Exception as e:
        print(f"  CelebA unavailable ({e}). Generating synthetic no_hair images …")

    if not celeba_ok:
        # Fallback: generate synthetic near-uniform head-shaped images
        # to represent a bald head (dark oval on light background)
        print("  Generating synthetic bald-head images as fallback …")
        no_hair_dir = ATTR_DIR / "no_hair"
        rng = np.random.default_rng(seed=42)
        for i in range(200):
            img = np.full((IMG_SIZE, IMG_SIZE), 220, dtype=np.uint8)
            # Draw a dark oval in the center (head shape)
            cy, cx = IMG_SIZE // 2, IMG_SIZE // 2
            ry, rx = int(IMG_SIZE * 0.38), int(IMG_SIZE * 0.30)
            for y in range(IMG_SIZE):
                for x in range(IMG_SIZE):
                    if ((y - cy) ** 2) / (ry ** 2) + ((x - cx) ** 2) / (rx ** 2) <= 1:
                        img[y, x] = int(rng.integers(120, 180))
            noise = rng.integers(-15, 15, (IMG_SIZE, IMG_SIZE))
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            Image.fromarray(img, mode="L").save(no_hair_dir / f"no_hair_{i+1:04d}.png")
        print(f"  Saved 200 synthetic images → attributes/no_hair/")

    # ---- Multiple persons composites ----
    if lfw is not None:
        generate_multiple(lfw, ATTR_DIR)
    else:
        # Fallback: use Olivetti pairs
        try:
            from sklearn.datasets import fetch_olivetti_faces
            data = fetch_olivetti_faces()
            imgs = data.images
            multi_dir = ATTR_DIR / "multiple"
            half = IMG_SIZE // 2
            count = 0
            rng = np.random.default_rng(seed=3)
            idx = rng.permutation(len(imgs))
            i = 0
            while count < MAX_PER_CLASS and i + 1 < len(idx):
                left  = (imgs[idx[i]]   * 255).astype(np.uint8)
                right = (imgs[idx[i+1]] * 255).astype(np.uint8)
                canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
                lp = Image.fromarray(left,  mode="L").resize((half, IMG_SIZE), Image.Resampling.LANCZOS)
                rp = Image.fromarray(right, mode="L").resize((half, IMG_SIZE), Image.Resampling.LANCZOS)
                canvas[:, :half] = np.array(lp)
                canvas[:, half:] = np.array(rp)
                Image.fromarray(canvas, mode="L").save(multi_dir / f"multiple_{count+1:04d}.png")
                count += 1
                i += 2
            print(f"  Saved {count} images → attributes/multiple/ (Olivetti pairs)")
        except Exception as e:
            print(f"  Could not generate multiple images: {e}")

    # ---- Synthetic backgrounds ----
    generate_backgrounds(BASE_DIR)

    # ---- Summary ----
    print("\n" + "=" * 55)
    print("Dataset Ready!")
    print("=" * 55)
    print("\nAttribute classes:")
    for cls in ATTR_CLASSES:
        d = ATTR_DIR / cls
        n = len(list(d.glob("*.png"))) if d.exists() else 0
        print(f"  {cls:12s}: {n:4d} images")
    print("\nStage A:")
    for sub in ["person", "no_person"]:
        d = BASE_DIR / "stage_a" / sub
        n = len(list(d.glob("*.png"))) if d.exists() else 0
        print(f"  {sub:12s}: {n:4d} images")
    print("\nNext step:")
    print("  python3 C_preprocess_and_augment.py --dataset_dir dataset --output_dir processed --augment_train --augmentations 5")


if __name__ == "__main__":
    main()
