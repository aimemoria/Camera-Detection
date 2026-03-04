#!/usr/bin/env python3
"""
Download and generate Stage A dataset for face detection.

Stage A classes:
  person     — LFW (1000) + Olivetti AT&T Faces (400)
  no_person  — synthetic backgrounds (500) + CIFAR-10 (1000)

CIFAR-10 is loaded from the local Keras cache (~/.keras/datasets/) using
plain pickle — no TensorFlow import, no GPU initialisation.

Usage:
    python3 download_larger_dataset.py
"""

import numpy as np
from PIL import Image
from pathlib import Path
import shutil

BASE_DIR      = Path("dataset")
STAGE_A_DIR   = BASE_DIR / "stage_a"
PERSON_DIR    = STAGE_A_DIR / "person"
NO_PERSON_DIR = STAGE_A_DIR / "no_person"

IMG_SIZE      = 96
MAX_PERSONS   = 1000  # LFW person images
MAX_UTK       = 400   # Olivetti AT&T Faces (all 400 available)
MAX_BG        = 500   # synthetic no_person images
MAX_CIFAR_BG  = 1000  # CIFAR-10 real-world no_person images (loaded via pickle)


def save_gray(arr: np.ndarray, path: Path):
    img = Image.fromarray(arr.astype(np.uint8), mode="L")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    img.save(path)


def download_lfw_persons():
    from sklearn.datasets import fetch_lfw_people

    print("  Downloading LFW faces (min_faces_per_person=1) ...")
    lfw = fetch_lfw_people(min_faces_per_person=1, resize=0.5)
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
    patterns = 6

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

    print(f"  Generated {n} synthetic background images → stage_a/no_person/")
    return n


def download_cifar10_backgrounds():
    """
    Load CIFAR-10 from the local Keras pickle cache and extract non-person classes
    as real background images.  Uses plain pickle — no TensorFlow import.

    Uses classes: 0=airplane, 1=automobile, 8=ship, 9=truck
    Images are 32x32 RGB — upsampled to 96x96 grayscale via LANCZOS.
    The soft upsampled textures resemble low-quality embedded camera output,
    which is beneficial for robustness training.

    If the cache is missing, downloads CIFAR-10 via urllib (~162 MB).
    """
    import pickle
    import urllib.request
    import tarfile

    cifar_dir = Path.home() / '.keras/datasets/cifar-10-batches-py-target/cifar-10-batches-py'

    if not cifar_dir.exists():
        print("  CIFAR-10 not cached — downloading via urllib (~162 MB) ...")
        url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        tmp = Path('/tmp/cifar10.tar.gz')
        urllib.request.urlretrieve(url, tmp)
        cifar_dir.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tmp) as tar:
            tar.extractall(cifar_dir.parent.parent)
        tmp.unlink(missing_ok=True)
        print("  CIFAR-10 downloaded and extracted.")
    else:
        print("  Loading CIFAR-10 from local cache ...")

    batch_files = ['data_batch_1', 'data_batch_2', 'data_batch_3',
                   'data_batch_4', 'data_batch_5', 'test_batch']
    all_data, all_labels = [], []
    for name in batch_files:
        with open(cifar_dir / name, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
        all_data.append(batch[b'data'])
        all_labels.extend(batch[b'labels'])

    # Reshape (N, 3072) → (N, 32, 32, 3)
    x_all = np.concatenate(all_data, axis=0).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_all = np.array(all_labels)

    # Non-person CIFAR-10 classes only
    non_person_classes = {0, 1, 8, 9}  # airplane, automobile, ship, truck
    mask = np.isin(y_all, list(non_person_classes))
    x_bg = x_all[mask]
    print(f"  CIFAR-10 non-face images available: {len(x_bg)}")

    rng = np.random.default_rng(seed=123)
    indices = rng.choice(len(x_bg), size=MAX_CIFAR_BG, replace=False)

    for i, idx in enumerate(indices):
        img_rgb = Image.fromarray(x_bg[idx], mode='RGB')
        img_gray = img_rgb.convert('L')
        img_resized = img_gray.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        img_resized.save(NO_PERSON_DIR / f"cifar_bg_{i+1:04d}.png")

    print(f"  Saved {MAX_CIFAR_BG} CIFAR-10 background images → stage_a/no_person/")
    return MAX_CIFAR_BG


def download_olivetti_persons():
    """
    Download Olivetti AT&T Faces via sklearn.

    400 images of 40 distinct subjects (10 photos each):
    - Taken at different times with varying lighting conditions
    - Different facial expressions (open/closed eyes, smiling/not smiling)
    - Glasses / no glasses variation
    - 64x64 grayscale — upsampled to 96x96

    Supplements LFW by adding subjects photographed in a structured
    study environment (different from press/celebrity photos in LFW).
    """
    from sklearn.datasets import fetch_olivetti_faces

    print("  Downloading Olivetti AT&T Faces (sklearn) ...")
    olivetti = fetch_olivetti_faces(shuffle=True, random_state=42)
    images = olivetti.images  # (400, 64, 64) float32 [0,1]
    print(f"  Olivetti loaded: {len(images)} images, {len(np.unique(olivetti.target))} subjects")

    count = min(len(images), MAX_UTK)
    for i in range(count):
        arr = (images[i] * 255).astype(np.uint8)
        save_gray(arr, PERSON_DIR / f"olivetti_{i+1:04d}.png")

    print(f"  Saved {count} Olivetti images → stage_a/person/")
    return count



def main():
    print("=" * 55)
    print("  Stage A Dataset Setup — Face Detection Only")
    print("=" * 55)

    if BASE_DIR.exists():
        shutil.rmtree(BASE_DIR)
    PERSON_DIR.mkdir(parents=True, exist_ok=True)
    NO_PERSON_DIR.mkdir(parents=True, exist_ok=True)

    persons      = download_lfw_persons()
    utk_count    = download_olivetti_persons()
    bg_count     = generate_backgrounds()
    cifar_count  = download_cifar10_backgrounds()

    person_total    = persons + utk_count
    no_person_total = bg_count + cifar_count

    print("\n" + "=" * 55)
    print("Dataset Ready!")
    print("=" * 55)
    print(f"  person:    {person_total} images ({persons} LFW + {utk_count} Olivetti)")
    print(f"  no_person: {no_person_total} images ({bg_count} synthetic + {cifar_count} CIFAR-10)")
    print(f"  Total:     {person_total + no_person_total} images")
    print("\nNext steps:")
    print("  python3 C_preprocess_and_augment.py --dataset_dir dataset --output_dir processed --augment_train --augmentations 6")
    print("  python3 E_train_model.py --data_dir processed --output_dir models")
    print("  python3 F_quantize_model.py --model_dir models --data_dir processed --validate")


if __name__ == "__main__":
    main()
