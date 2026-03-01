# TinyML Person Detection Dataset Design

## Hardware Constraints
- **Arduino Nano 33 Sense**: 256KB RAM, 1MB Flash
- **Target Resolution**: 96x96 RGB (preferred) or 64x64 grayscale (minimal)
- **Model Size Target**: ≤100KB after INT8 quantization

---

## Dataset Structure

```
dataset/
├── train/
│   ├── person/          # Images with people
│   └── no_person/       # Background scenes without people
├── validation/
│   ├── person/
│   └── no_person/
└── test/
    ├── person/
    └── no_person/
```

---

## Minimum Dataset Size

### Recommended Sizes:
- **Training**: 2,000-3,000 images per class (4,000-6,000 total)
- **Validation**: 400-600 images per class (800-1,200 total)
- **Test**: 400-600 images per class (800-1,200 total)

### Minimal Viable:
- **Training**: 1,000 images per class (2,000 total)
- **Validation**: 200 images per class (400 total)
- **Test**: 200 images per class (400 total)

**Rationale**: Small models overfit easily. Need sufficient variation to generalize on tiny parameter budgets.

---

## Resolution Recommendation

### Option 1: 96x96 RGB (Recommended)
- **Input Shape**: (96, 96, 3)
- **Memory**: 27,648 bytes per image
- **Pros**: Better accuracy, color information
- **Cons**: Higher memory usage

### Option 2: 64x64 Grayscale (Minimal)
- **Input Shape**: (64, 64, 1)
- **Memory**: 4,096 bytes per image
- **Pros**: 6.75x less memory, faster inference
- **Cons**: Loss of color information

### Recommendation: **Start with 96x96 RGB**, fallback to grayscale if OOM issues arise.

---

## Data Collection Strategy

### Class 1: Person

#### Lighting Variation
- **Indoor**: Office, home, dim lighting, bright lighting
- **Outdoor**: Sunny, cloudy, shade, golden hour, dusk
- **Artificial**: Fluorescent, LED, warm/cool tones
- **Target**: 30% indoor, 40% outdoor, 30% mixed

#### Distance Variation
- **Close**: 0.5-2 meters (upper body fills 60-80% of frame)
- **Medium**: 2-5 meters (full body fills 40-70% of frame)
- **Far**: 5-10 meters (person fills 20-40% of frame)
- **Target**: 40% medium, 30% close, 30% far

#### Orientation Variation
- **Frontal**: Facing camera directly
- **Profile**: Side view (90°)
- **Back**: Rear view (180°)
- **Oblique**: 45° angles
- **Target**: 40% frontal, 30% oblique, 20% profile, 10% back

#### Background Variation
- Indoor: Walls, furniture, doors, windows
- Outdoor: Trees, buildings, streets, parks
- Complex: Crowds, patterns, similar shapes
- **Target**: Even distribution across contexts

#### Demographic Diversity
- Age groups: Children, adults, elderly
- Body types: Various heights, builds
- Clothing: Casual, formal, seasonal, uniforms
- Accessories: Hats, bags, glasses

### Class 2: No Person

#### Background Types
- **Indoor**: Empty rooms, furniture, appliances, walls
- **Outdoor**: Landscapes, buildings, streets, sky
- **Animals**: Dogs, cats, birds (potential false positives)
- **Objects**: Cars, bikes, trash cans, poles, trees
- **Target**: High diversity to prevent background overfitting

#### Hard Negatives
- Mannequins or statues (human-shaped but not people)
- Posters or paintings of people
- Shadows that resemble human silhouettes
- Vertical objects (poles, trees) that might be confused with people

---

## Data Capture Guidelines

### Camera Setup
- **Device**: Smartphone or webcam with 1-5MP resolution
- **Capture**: High resolution, then downscale during preprocessing
- **Format**: JPEG or PNG
- **Naming**: Descriptive (e.g., `person_outdoor_medium_01234.jpg`)

### Quality Requirements
- **No blur**: Ensure sharp focus
- **Proper exposure**: Not overexposed or underexposed
- **No corruption**: No file errors
- **Aspect ratio**: Try to maintain square-ish (will be cropped to square)

### Ethical Considerations
- Obtain consent for identifiable individuals
- Avoid capturing sensitive locations
- Remove metadata (EXIF) containing GPS/personal info
- Consider privacy laws (GDPR, CCPA)

---

## Augmentation Strategy (Applied During Training)

### Geometric Augmentations
- **Horizontal flip**: 50% probability
- **Rotation**: ±10° random
- **Zoom**: 0.9-1.1x random scale
- **Shift**: ±10% horizontal/vertical

### Photometric Augmentations
- **Brightness**: ±20% random adjustment
- **Contrast**: 0.8-1.2x random multiplier
- **Saturation**: 0.8-1.2x (RGB only)
- **Noise**: Gaussian noise, σ=0.02

### Rationale
Augmentation artificially expands dataset and improves model robustness to real-world variations.

---

## Memory Budget Analysis

### During Training (GPU)
- Batch size: 32
- 96x96 RGB: 32 × 27,648 = 884,736 bytes (~0.84 MB per batch)
- Model + optimizer: ~50-100 MB
- **Requirement**: 4GB+ GPU RAM (easily met)

### During Inference (Arduino)
- Single image: 96×96×3 = 27,648 bytes (~27 KB)
- Model weights: ~80-100 KB
- Tensor arena: ~100-150 KB
- **Total RAM**: ~200-250 KB (fits in 256 KB)

### Flash Memory
- Model binary: ~80-100 KB
- Arduino firmware: ~50-80 KB
- **Total Flash**: ~130-180 KB (well under 1 MB)

---

## Pre-collection Checklist

- [ ] Create dataset folder structure
- [ ] Prepare camera device (smartphone/webcam)
- [ ] Plan capture locations (indoor/outdoor)
- [ ] Recruit subjects (if needed) and obtain consent
- [ ] Set naming convention
- [ ] Allocate 3-5 hours for initial capture session
- [ ] Target: 500-1000 images per session

---

## Quality Assurance

After collection, manually review:
1. **Mislabeled images**: Person in "no_person" or vice versa
2. **Corrupted files**: Cannot open or display
3. **Duplicate images**: Remove exact copies
4. **Poor quality**: Blur, extreme darkness, extreme brightness

Use scripts to:
- Check file integrity
- Remove duplicates (perceptual hashing)
- Verify image dimensions
- Balance class distribution

---

## Expected Timeline

- **Data collection**: 2-4 days (casual) or 1 day (intensive)
- **Data cleaning**: 0.5-1 day
- **Preprocessing setup**: 0.5 day
- **Total**: 3-6 days for robust dataset

---

## Summary

| Parameter | Recommended | Minimal |
|-----------|-------------|---------|
| **Resolution** | 96x96 RGB | 64x64 Grayscale |
| **Train Images** | 4,000-6,000 | 2,000 |
| **Val Images** | 800-1,200 | 400 |
| **Test Images** | 800-1,200 | 400 |
| **Lighting Variants** | 5+ conditions | 3 conditions |
| **Distance Variants** | 3 ranges | 2 ranges |
| **Orientation** | 4 angles | 2 angles |

**Next Step**: Collect images following this specification, then proceed to preprocessing pipeline.
