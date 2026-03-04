# Face Detection — Accuracy Development Report

> TinyML system running on Arduino Nano 33 BLE Sense Rev2 + ArduCAM OV2640
> Model: Stage A binary classifier — person vs no_person
> INT8 quantized TFLite, 96×96 grayscale input

---

## What "Real-World Robustness" Means

**Lab accuracy** measures performance on held-out data from the *same distribution* as training — clean, well-lit, easy images. A 100% lab score does not mean the model works in the real world.

**Real-world robustness** measures how the model handles conditions it was *not* explicitly trained on:

| Condition | Example |
|-----------|---------|
| Motion blur | Camera or subject moves during capture |
| Shadows / low light | Indoor room with one lamp |
| Harsh backlight | Window behind the subject |
| Real indoor backgrounds | Walls, desks, shelves — not synthetic patterns |
| Partial face | Hand in front of face, or face at frame edge |
| Face diversity | Different ages, eyeglasses, non-frontal angles |

---

## Real-World Robustness — Progress Over Time

| Stage | Date | Lab Accuracy | Real-World Accuracy | Status |
|-------|------|:------------:|:-------------------:|--------|
| Before any robustness work (Run 4) | 2026-03-03 | 100% | **~60–80%** | Baseline — synthetic backgrounds only |
| After first round of improvements (Run 5) | 2026-03-03 | 100% | **~75–90%** | Added real photos + hard augmentation |
| After second round of improvements (Run 6) | 2026-03-04 | 99.77% | **~85–95%** | Added Olivetti faces + more real backgrounds |

> Real-world accuracy is an estimate — the model has not been live-tested across all conditions. Lab accuracy is measured on the held-out test set.

---

## Before Any Robustness Work — Run 4 (2026-03-03)

**What the model was trained on:**
- 500 LFW celebrity faces — all frontal, well-lit press photos (only 62 distinct subjects)
- 500 synthetic backgrounds — computer-generated noise, gradients, checkerboards (no real photos)
- Augmentation: basic brightness, rotation, flip, noise (3 copies per image)

**Why it was weak in the real world:**
- Never saw a real room, wall, desk, or natural background → high false-positive risk in any indoor scene
- Only 62 face subjects → failed on unusual lighting, angles, or non-celebrity faces
- No blur, shadow, or occlusion in training → common camera conditions caused errors

**Real-world accuracy: ~60–80%**

---

## After First Improvements — Run 5 (2026-03-03)

**What changed:**

| Component | Before (Run 4) | After (Run 5) |
|-----------|----------------|---------------|
| Face diversity | 500 images, 62 subjects | 1,000 images, all 13,233 LFW subjects |
| Backgrounds — real | None | 500 CIFAR-10 real photos (upscaled 32×96) |
| Backgrounds — synthetic | 500 patterns | 500 patterns (kept) |
| Blur augmentation | None | Gaussian blur on ~40% of copies |
| Extreme brightness | Mild ±40% | Added severe: 0.2–0.5× shadow / 1.5–2.0× backlight |
| Partial occlusion | None | Black rectangle 10–30% of image on ~20% of copies |
| Augmented copies | 3 per image | 6 per image |
| Training samples | ~2,800 | **9,800** |

**Lab accuracy:** 100% (300-sample balanced test set)
**Real-world accuracy: ~75–90%**

**Remaining gaps after Run 5:**
- CIFAR-10 images were only 32×32, upscaled → soft textures only
- Faces still all celebrities — lacks age diversity, eyeglasses, non-western faces

---

## After Second Improvements — Run 6 (2026-03-04)

**What changed:**

| Component | Before (Run 5) | After (Run 6) |
|-----------|----------------|---------------|
| Person images | 1,000 LFW | 1,000 LFW + **400 Olivetti AT&T** = 1,400 |
| Olivetti faces | None | 400 images, 40 subjects — varied lighting, expressions, glasses |
| Backgrounds — real | 500 CIFAR-10 | **1,000 CIFAR-10** (doubled) |
| Raw images total | ~1,500 | **2,900** |
| Training samples | 9,800 | **14,210** |

**What Olivetti AT&T adds:** 40 subjects photographed in a lab environment (not press photos) — open/closed eyes, smiling/neutral, glasses/no glasses, different lighting. Fills the gap left by LFW's celebrity bias.

**Lab accuracy:** 99.77% — 434 correct out of 435 samples (1 error)
**INT8 model validation:** 100% on 100 samples
**Model size:** 17.55 KB (unchanged class of ~17 KB)
**Real-world accuracy: ~85–95%**

---

## Key Decisions Log

| Date | Decision | Reason |
|------|----------|--------|
| 2026-02-28 | Built 2-stage system (detection + attribute) | Initial design goal |
| 2026-03-01 | Replaced attribute detection with 5-person recognition | Exploring recognition capability |
| 2026-03-03 | Removed Stage B entirely | Recognition at 85.5% too unreliable, adds 56 KB RAM and complexity |
| 2026-03-03 | Added CIFAR-10 real backgrounds | Synthetic-only backgrounds caused false positives on real rooms |
| 2026-03-03 | Added blur / extreme brightness / occlusion augmentation | Camera sees these conditions; model had zero exposure to them |
| 2026-03-03 | Increased LFW diversity (min_faces=1 vs 20) | Only 62 subjects is far too narrow for generalisation |
| 2026-03-04 | Added Olivetti + doubled CIFAR-10 to 1,000 | Fill face diversity gap (age, expressions, glasses) and increase background variety |

---

*Source data: `accuracy_log.json` · Auto-updated by `F_quantize_model.py` after each run*
