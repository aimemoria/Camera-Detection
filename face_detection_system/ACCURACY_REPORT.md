# Face Detection — Accuracy Development Report

> TinyML system running on Arduino Nano 33 BLE Sense Rev2 + ArduCAM OV2640
> Model: INT8 quantized TFLite, 96×96 grayscale input

---

## Summary

| Run | Date | System | Test Accuracy | Model Size |
|-----|------|--------|:-------------:|:----------:|
| 1 | 2026-02-28 | Stage A — Person Detection (initial) | **100.00%** | — |
| 2 | 2026-02-28 | Stage B v1 — Attribute Detection (6 classes) | **98.81%** | — |
| 3 | 2026-03-01 | Stage B v2 — 5-Person Recognition | **85.52%** | — |
| 4 | 2026-03-03 | Stage A only — Detection, balanced dataset | **100.00%** | 17.52 KB |
| 5 | 2026-03-03 | Stage A — Real-world robustness improvements | **100.00%** | 17.52 KB |

---

## Run 1 — 2026-02-28 · Initial Face Detection

**Model:** Stage A (binary: person / no_person)

**Dataset:**
- 500 LFW face images (`min_faces_per_person=20` — frequent subjects only)
- 200 synthetic background images (noise, gradients)

**Augmentation:** Basic (brightness, contrast, rotation, flip)

**Result:** 100.00% — 201 test samples, 0 errors

**Limitation:** Test set was easy — same narrow distribution as training. Synthetic backgrounds looked nothing like a real room. Real-world accuracy unknown.

---

## Run 2 — 2026-02-28 · Attribute Detection (6 Classes)

**Model:** Stage B v1 (6-class: grief, hair, happy, multiple, no_hair, sad)

**Dataset:** Custom attribute-labeled face images

**Result:** 98.81% — 168 test samples, 2 errors

**Note:** This was a separate Stage B classifier running after Stage A detected a face. High accuracy on the attribute task.

---

## Run 3 — 2026-03-01 · 5-Person Face Recognition

**Model:** Stage B v2 (5-class: Colin Powell, Donald Rumsfeld, George W Bush, Gerhard Schroeder, Tony Blair)

**Dataset:** LFW images filtered to 5 specific people

**Result:** 85.52% — 152 test samples, 22 errors

**What happened:** Accuracy dropped significantly. Recognising specific individuals is a much harder problem than detecting a face vs. no face — more classes, less data per class, higher inter-class similarity.

**Decision:** Removed Stage B entirely. System reverted to detection-only.

---

## Run 4 — 2026-03-03 · Detection Only, Balanced Dataset

**Model:** Stage A only (binary: person / no_person)
**Changes from Run 1:**
- Stage B removed entirely
- Dataset rebalanced: 500 person + 500 no_person (was 500+200)
- Augmentations increased to 5 per image

**Result:** 100.00% — 201 test samples, 0 errors
**Model size:** 17.52 KB INT8 (12x compression from 211 KB float32)

**Limitation:** Still trained on synthetic backgrounds only. Test accuracy of 100% does not reflect real-world performance with actual camera images.

---

## Run 5 — 2026-03-03 · Real-World Robustness Improvements

**Model:** Stage A (binary: person / no_person)
**Goal:** Improve performance on real camera images — blur, shadows, real backgrounds, partial faces

### What Changed

**Dataset (before → after):**

| Component | Before | After |
|-----------|--------|-------|
| Face images | 500 (frequent LFW subjects only) | 1000 (all 13,233 LFW subjects) |
| Backgrounds — synthetic | 500 patterns | 500 patterns (kept) |
| Backgrounds — real | None | 500 CIFAR-10 real scenes (airplane, automobile, ship, truck) |
| **Total images** | **1000** | **2000** |

**Augmentation (before → after):**

| Augmentation | Before | After |
|--------------|--------|-------|
| Brightness (mild) | ✓ | ✓ |
| Contrast | ✓ | ✓ |
| Rotation ±20° | ✓ | ✓ |
| Shift ±10% | ✓ | ✓ |
| Zoom 0.9–1.1× | ✓ | ✓ |
| Horizontal flip | ✓ | ✓ |
| Gaussian noise | ✓ | ✓ |
| **Gaussian blur** (defocus/motion) | ✗ | **✓ ~40% of copies** |
| **Extreme brightness** (shadow/backlight) | ✗ | **✓ ~25% of copies** |
| **Partial occlusion** (10–30% black rect) | ✗ | **✓ ~20% of copies** |
| Augmented copies per image | 3 | **6** |

**Training scale (before → after):**

| Metric | Before (Run 4) | After (Run 5) |
|--------|---------------|--------------|
| Training samples | ~2,800 | **9,800** |
| Validation samples | ~300 | 300 |
| Test samples | ~201 | **300** (150 per class, balanced) |

**Result:** 100.00% on the new harder test set — 300 samples, 0 errors
**Model size:** 17.52 KB (unchanged — no architecture changes)

### Why This Matters

The test set now contains:
- Faces from diverse LFW subjects (varied angles, ages, ethnicities, lighting)
- Real CIFAR-10 background images (not just synthetic patterns)

A 100% score here is more meaningful than previous 100% scores — the model is being evaluated on a harder distribution that better reflects real camera conditions.

---

## Estimated Real-World Performance (After Run 5)

| Condition | Estimated Accuracy |
|-----------|:-----------------:|
| Clear frontal face, good light | ~92–97% |
| Mild shadow or side angle | ~80–90% |
| Blurry (motion/defocus) | ~70–85% |
| Dark room / low light | ~60–75% |
| Real indoor background (no face) correctly rejected | ~85–95% |

> These are estimates. Actual measurement requires capturing real ArduCam images and labelling them.

---

## Key Decisions Log

| Date | Decision | Reason |
|------|----------|--------|
| 2026-02-28 | Built 2-stage system (detection + attribute) | Initial design |
| 2026-03-01 | Replaced attribute detection with 5-person recognition | Exploring recognition |
| 2026-03-03 | Removed Stage B entirely | Recognition accuracy too low (85.5%), adds complexity and 56 KB RAM |
| 2026-03-03 | Added CIFAR-10 real backgrounds | Synthetic-only backgrounds caused false positives with real rooms |
| 2026-03-03 | Added blur, extreme brightness, occlusion augmentation | Model never saw these conditions — common with ArduCam |
| 2026-03-03 | Increased LFW diversity (min_faces=1 vs 20) | More varied faces = better generalisation |

---

*Source data: `accuracy_log.json` · Auto-updated by `F_quantize_model.py` after each training run*
