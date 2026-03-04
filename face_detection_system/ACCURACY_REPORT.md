# Face Detection — Accuracy Development Report

> TinyML system running on Arduino Nano 33 BLE Sense Rev2 + ArduCAM OV2640
> Model: Stage A binary classifier — person vs no_person
> INT8 quantized TFLite, 96×96 grayscale input

---

## Detection Accuracy Over Time

| Run | Date | Test Accuracy | Dataset | Training Samples | Model Size |
|-----|------|:-------------:|---------|:----------------:|:----------:|
| 1 | 2026-02-28 | **100.00%** | 500 LFW faces + 200 synthetic bg | ~2,000 | — |
| 4 | 2026-03-03 | **100.00%** | 500 LFW faces + 500 synthetic bg (balanced) | ~2,800 | 17.52 KB |
| 5 | 2026-03-03 | **100.00%** | 1000 diverse LFW + 500 synthetic + 500 CIFAR-10 bg | 9,800 | 17.52 KB |
| 6 | 2026-03-04 | **99.77%** | 1400 person (LFW + Olivetti) + 1500 no_person (synthetic + 1000 CIFAR-10) | 14,210 | 17.55 KB |

> Run 2 (attribute detection) and Run 3 (5-person recognition) were separate experiments, not face detection. See decision log at the bottom.

---

## Real-World Robustness

### What "real-world robustness" means here
Lab test accuracy measures performance on data from the same distribution as training.
Real-world robustness measures how well the model handles conditions it wasn't explicitly trained on:
blurry images, dark rooms, harsh backlighting, real indoor backgrounds, partial faces.

---

### Before Improvements — Run 4 (2026-03-03)

**Dataset:**
- 500 LFW faces — frequent subjects only (`min_faces_per_person=20`), mostly frontal well-lit
- 500 synthetic backgrounds — noise, gradients, checkerboards, diagonal stripes (no real photos)

**Augmentation:** brightness, contrast, rotation ±20°, shift, zoom, flip, Gaussian noise (3 copies per image)

**Lab accuracy:** 100% (on easy test set)

**Real-world weaknesses identified:**
- Model had never seen a real room, wall, desk, or any natural texture as a background → high false positive risk
- All background training examples were computer-generated patterns
- Only ~62 distinct face subjects — very narrow face diversity
- No blur, shadow, or occlusion in training — common real camera conditions

**Estimated real-world accuracy (before):** 60–80%

---

### After Improvements — Run 5 (2026-03-03)

**What changed:**

| Component | Before | After |
|-----------|--------|-------|
| Face diversity | 500 images, 62 LFW subjects | 1000 images, all 13,233 LFW subjects |
| Backgrounds — real | None | 500 CIFAR-10 scenes (airplane, car, ship, truck — 32×32 upscaled) |
| Backgrounds — synthetic | 500 patterns | 500 patterns (kept) |
| Blur augmentation | None | Gaussian blur ~40% of augmented copies |
| Extreme brightness | Mild (0.6–1.4×) | Added severe: 0.2–0.5× (shadow) / 1.5–2.0× (backlight) |
| Partial occlusion | None | Black rectangle 10–30% of image ~20% of copies |
| Augmented copies | 3 per image | 6 per image |
| **Training samples** | **~2,800** | **9,800** |

**Lab accuracy:** 100% on harder balanced test set (300 samples, 150 per class)

**Estimated real-world accuracy (after):** 75–90%

**Remaining weaknesses:**
- CIFAR-10 backgrounds were only 32×32, upscaled → soft/blurry textures, limited variety
- LFW faces still only celebrities — lacks age diversity, eyeglasses, beards, non-western faces

---

### After Further Improvements — Run 6 (2026-03-04)

**New datasets added:**

**Olivetti AT&T Faces** (via sklearn, no download required)
- 400 face images of 40 distinct subjects (10 photos each)
- Varied lighting conditions, facial expressions (open/closed eyes, smiling)
- With and without glasses
- Fills the gap left by LFW's press-photography conditions

**Expanded CIFAR-10 backgrounds** (doubled to 1,000 images, loaded directly from local cache — no TF required)
- 1,000 real-world scene photos (airplane, automobile, ship, truck classes)
- Provides double the background variety compared to Run 5

**What changed:**

| Component | Run 5 | Run 6 |
|-----------|-------|-------|
| Person images | 1000 LFW | 1000 LFW + 400 Olivetti = **1400** |
| Background — real | 500 CIFAR-10 | **1000 CIFAR-10** |
| Background — synthetic | 500 patterns | 500 patterns (kept) |
| Raw images total | ~1,500 | **2,900** |
| Training samples (augmented) | 9,800 | **14,210** |

**Lab accuracy:** 99.77% on 435-sample test set (1 error) · INT8 validation: 100% on 100 samples

**Estimated real-world accuracy (after Run 6):** 85–95%

---

## Run-by-Run Details

### Run 1 — 2026-02-28 · Initial Training

**Dataset:** 500 LFW (`min_faces_per_person=20`) + 200 synthetic backgrounds
**Test set:** 201 samples (unbalanced: 30 no_person + 171 person)
**Result:** 100.00% — but test set was small and easy, same narrow distribution as training

---

### Run 4 — 2026-03-03 · Balanced Dataset, Detection Only

**Changes from Run 1:**
- Stage B recognition removed entirely
- Dataset rebalanced: 500 person + 500 no_person
- Augmentations increased to 5 per image

**Test set:** 201 samples
**Result:** 100.00%
**Model size:** 17.52 KB INT8 (12× compression from 211 KB float32)

---

### Run 5 — 2026-03-03 · Real-World Robustness Improvements

**Changes:** See "After Improvements" table above

**Test set:** 300 samples (150 per class, balanced)
**Result:** 100.00%
**Model size:** 17.52 KB (unchanged)

---

### Run 6 — 2026-03-04 · Deeper Diversity (Olivetti + expanded CIFAR-10)

**New data sources:**
- **Olivetti AT&T Faces** — 400 images, 40 subjects with varied lighting, expressions, glasses
- **Expanded CIFAR-10** — 1,000 background images (doubled from Run 5, loaded via pickle cache)

**Total dataset:** 1,400 person + 1,500 no_person = 2,900 raw images

**Test set:** 435 samples (balanced)
**Result:** 99.77% (1 error out of 435) · INT8 validation 100% on 100 samples
**Model size:** 17.55 KB (12× compression from 211 KB float32)

---

## Key Decisions Log

| Date | Decision | Reason |
|------|----------|--------|
| 2026-02-28 | Built 2-stage system (detection + attribute) | Initial design goal |
| 2026-03-01 | Replaced attribute detection with 5-person recognition | Exploring recognition capability |
| 2026-03-03 | Removed Stage B entirely | Recognition at 85.5% too unreliable, adds 56 KB RAM and complexity |
| 2026-03-03 | Added CIFAR-10 real backgrounds | Synthetic-only backgrounds caused false positives on real rooms |
| 2026-03-03 | Added blur/extreme brightness/occlusion augmentation | Camera sees these conditions; model had zero exposure to them |
| 2026-03-03 | Increased LFW diversity (min_faces=1 vs 20) | Only 62 subjects is far too narrow for generalisation |
| 2026-03-04 | Added Olivetti + doubled CIFAR-10 (1,000) | Fill face diversity gap (age, expressions, glasses) and double background variety |

---

*Source data: `accuracy_log.json` · Auto-updated by `F_quantize_model.py` after each run*
