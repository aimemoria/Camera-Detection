# Mathematical Explanation: Why Person Detection Uses 2D Convolution

## Table of Contents
1. [Fundamental Tensor Concepts](#fundamental-tensor-concepts)
2. [Why 2D Convolution for Images](#why-2d-convolution-for-images)
3. [Tensor Dimension Analysis](#tensor-dimension-analysis)
4. [Comparison: Speech vs Vision](#comparison-speech-vs-vision)
5. [Computational Complexity](#computational-complexity)

---

## Fundamental Tensor Concepts

### Tensor Rank and Shape

A **tensor** is a multi-dimensional array. Its **rank** is the number of dimensions.

| Rank | Name | Example | Shape |
|------|------|---------|-------|
| 0 | Scalar | Temperature: 72.5°F | `()` |
| 1 | Vector | Audio sample: [0.1, 0.3, -0.2, ...] | `(T,)` |
| 2 | Matrix | Grayscale image | `(H, W)` |
| 3 | 3D Tensor | RGB image | `(H, W, C)` |
| 4 | 4D Tensor | Batch of RGB images | `(B, H, W, C)` |

### Image Representation

An **RGB image** is a 3D tensor:

$$
\text{Image} \in \mathbb{R}^{H \times W \times C}
$$

Where:
- $H$ = height (pixels)
- $W$ = width (pixels)
- $C$ = channels (3 for RGB: Red, Green, Blue)

Example: A 96×96 RGB image has shape $(96, 96, 3)$, containing $96 \times 96 \times 3 = 27{,}648$ values.

---

## Why 2D Convolution for Images

### Spatial Structure is Key

Images have **spatial structure**: nearby pixels are correlated. A person's eye is close to their nose. A car's wheel is near its body. This **local correlation** is critical for understanding visual content.

### 1D vs 2D Convolution

#### 1D Convolution (For Time Series)
Used for **sequential data** like audio, speech, or sensor data:

$$
(f * g)[n] = \sum_{m=-\infty}^{\infty} f[m] \cdot g[n - m]
$$

- Operates along **one axis** (time)
- Example: Sliding a 1D filter over audio waveform

**Input**: Audio signal $\in \mathbb{R}^{T}$ (T timesteps)  
**Filter**: Weights $\in \mathbb{R}^{K}$ (K kernel size)  
**Output**: Feature map $\in \mathbb{R}^{T'}$

#### 2D Convolution (For Images)
Used for **grid-structured data** like images:

$$
(I * K)[i, j] = \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} I[i+m, j+n] \cdot K[m, n]
$$

Where:
- $I$ is the input image
- $K$ is the 2D kernel (filter)
- $(i, j)$ are spatial coordinates
- $(M, N)$ are kernel dimensions

**Input**: Image $\in \mathbb{R}^{H \times W \times C}$  
**Filter**: Weights $\in \mathbb{R}^{F_H \times F_W \times C}$ (e.g., 3×3×3)  
**Output**: Feature map $\in \mathbb{R}^{H' \times W' \times C'}$

### Why 2D for Person Detection?

**Person detection requires spatial reasoning:**

1. **Vertical structure**: A person's head is above their shoulders, above their torso.
2. **Horizontal symmetry**: Arms on both sides, legs on both sides.
3. **Shape detection**: Rounded head, rectangular torso, cylindrical limbs.
4. **Context**: A person standing in front of a building (background vs foreground).

A **1D convolution** would destroy spatial relationships by flattening the image:

```
Original (2D):          Flattened (1D):
[person's head]         [pixel1, pixel2, pixel3, ..., pixel27648]
[   shoulders  ]        ↑
[     torso    ]        No spatial structure!
[     legs     ]
```

The model would lose the ability to recognize that "head above shoulders" is a person.

---

## Tensor Dimension Analysis

### Forward Pass Through a CNN Layer

Let's trace tensor dimensions through one Conv2D layer:

#### Input
- **Shape**: $(B, H, W, C_{\text{in}})$
- **Example**: $(1, 96, 96, 3)$ (one 96×96 RGB image)

#### Convolution Operation
- **Filter shape**: $(F_H, F_W, C_{\text{in}}, C_{\text{out}})$
- **Example**: $(3, 3, 3, 16)$ (16 filters of size 3×3×3)

For each output channel $c_{\text{out}}$:

$$
\text{Output}[i, j, c_{\text{out}}] = \sum_{m=0}^{F_H-1} \sum_{n=0}^{F_W-1} \sum_{c_{\text{in}}=0}^{C_{\text{in}}-1} \text{Input}[i+m, j+n, c_{\text{in}}] \cdot \text{Filter}[m, n, c_{\text{in}}, c_{\text{out}}] + \text{bias}[c_{\text{out}}]
$$

#### Output
- **Shape** (with padding='same'): $(B, H, W, C_{\text{out}})$
- **Example**: $(1, 96, 96, 16)$

#### After MaxPooling (2×2)
- **Shape**: $(B, H/2, W/2, C_{\text{out}})$
- **Example**: $(1, 48, 48, 16)$

### Full CNN Architecture Dimensions

For our TinyML person detector:

| Layer | Input Shape | Kernel | Stride | Output Shape | Parameters |
|-------|-------------|--------|--------|--------------|------------|
| Input | (1, 96, 96, 3) | - | - | (1, 96, 96, 3) | 0 |
| Conv2D-1 | (1, 96, 96, 3) | (3, 3, 3, 16) | 1 | (1, 96, 96, 16) | 448 |
| MaxPool2D | (1, 96, 96, 16) | (2, 2) | 2 | (1, 48, 48, 16) | 0 |
| Conv2D-2 | (1, 48, 48, 16) | (3, 3, 16, 32) | 1 | (1, 48, 48, 32) | 4,640 |
| MaxPool2D | (1, 48, 48, 32) | (2, 2) | 2 | (1, 24, 24, 32) | 0 |
| Conv2D-3 | (1, 24, 24, 32) | (3, 3, 32, 48) | 1 | (1, 24, 24, 48) | 13,872 |
| MaxPool2D | (1, 24, 24, 48) | (2, 2) | 2 | (1, 12, 12, 48) | 0 |
| Flatten | (1, 12, 12, 48) | - | - | (1, 6912) | 0 |
| Dense | (1, 6912) | - | - | (1, 128) | 884,864 |
| Dense | (1, 128) | - | - | (1, 2) | 258 |
| **Total** | | | | | **~904K params** |

**Problem**: 904K parameters × 1 byte (INT8) = **~884 KB** — TOO LARGE!

**Solution**: Use GlobalAveragePooling instead of Flatten + large Dense layer:

| Layer | Input Shape | Output Shape | Parameters |
|-------|-------------|--------------|------------|
| ... (same as above until MaxPool-3) | | | |
| GlobalAvgPool | (1, 12, 12, 48) | (1, 48) | 0 |
| Dense | (1, 48) | (1, 64) | 3,136 |
| Dense | (1, 64) | (1, 2) | 130 |
| **Total** | | | **~22K params** |

Now: 22K parameters × 1 byte = **~22 KB** — Fits in target! ✓

---

## Comparison: Speech vs Vision

### Speech Recognition (1D Time Series)

**Input**: Audio waveform or spectrogram  
**Structure**: **Temporal** (time-dependent)  
**Processing**: 1D convolution along time axis

#### Waveform Representation
$$
\text{Audio} \in \mathbb{R}^{T}
$$

Where $T$ is the number of time samples.

Example: 1 second of 16kHz audio = 16,000 samples.

#### 1D Convolution Filter
$$
\text{Filter} \in \mathbb{R}^{K}
$$

Slides along time axis to detect patterns like phonemes ("p", "a", "t").

#### Why 1D?
Speech is inherently **sequential**:
- "cat" ≠ "tac" (order matters)
- Phonemes follow each other in time
- No spatial relationships (no "up/down/left/right")

### Person Detection (2D Spatial)

**Input**: Image (RGB or grayscale)  
**Structure**: **Spatial** (2D grid)  
**Processing**: 2D convolution across height and width

#### Image Representation
$$
\text{Image} \in \mathbb{R}^{H \times W \times C}
$$

Example: 96×96 RGB image = 27,648 values.

#### 2D Convolution Filter
$$
\text{Filter} \in \mathbb{R}^{F_H \times F_W \times C}
$$

Slides across height and width to detect patterns like edges, shapes, textures.

#### Why 2D?
Images are inherently **spatial**:
- Objects have shapes (circles, rectangles)
- Spatial relationships matter (head above body)
- Context from surrounding pixels (person vs background)

### Computational Comparison

| Task | Input Dim | Conv Type | Typical Input | Params (Conv1) | MACs (Conv1) |
|------|-----------|-----------|---------------|----------------|--------------|
| **Speech** | 1D | Conv1D | (16000,) | $K \times C$ | $T \times K \times C$ |
| **Vision** | 2D | Conv2D | (96, 96, 3) | $F_H \times F_W \times C \times C'$ | $H \times W \times F_H \times F_W \times C \times C'$ |

**Example**: One Conv2D filter (3×3, 3→16 channels) on 96×96 image:
$$
\text{MACs} = 96 \times 96 \times 3 \times 3 \times 3 \times 16 = 39{,}813{,}120 \approx 40M
$$

**Speech** (Conv1D, 16kHz, 1 second, kernel=25):
$$
\text{MACs} = 16{,}000 \times 25 \times 1 = 400{,}000 = 0.4M
$$

**Person detection is ~100× more compute-intensive per layer than speech!**

---

## Computational Complexity

### Why Person Detection is Heavier

#### 1. Input Size
- **Speech**: 1D array, ~16K samples/sec → $O(T)$
- **Vision**: 2D grid, 96×96×3 = 27K pixels → $O(H \times W \times C)$

#### 2. Convolution Operations
- **Speech (1D Conv)**: $O(T \times K \times C)$
- **Vision (2D Conv)**: $O(H \times W \times F_H \times F_W \times C \times C')$

For 96×96 image, 3×3 kernel, 3→16 channels:
$$
96 \times 96 \times 3 \times 3 \times 3 \times 16 = 39.8 \text{ million operations}
$$

#### 3. Memory Bandwidth
**Image**:
- Load 96×96×3 = 27,648 bytes per frame
- At 10 FPS: 276,480 bytes/sec

**Speech**:
- Load 16,000 samples/sec = 16,000 bytes/sec (16kHz, 8-bit)

**Vision requires ~17× more memory bandwidth.**

#### 4. Feature Hierarchy
- **Speech**: Phonemes → words → sentences (mostly 1D temporal)
- **Vision**: Edges → shapes → parts → objects (2D spatial at every level)

Each level in vision requires 2D convolutions, compounding complexity.

### Optimizations for Microcontrollers

Given these constraints, TinyML uses:

1. **Aggressive quantization**: FP32 → INT8 (4× memory reduction)
2. **Smaller input**: 96×96 or 64×64 instead of 224×224
3. **Fewer filters**: 16, 32, 48 instead of 256, 512, 1024
4. **Shallow networks**: 2-3 Conv layers instead of 50+ (ResNet)
5. **Global pooling**: Instead of large fully-connected layers

---

## Summary

### Key Takeaways

| Aspect | Speech (1D) | Person Detection (2D) |
|--------|-------------|------------------------|
| **Input structure** | Temporal sequence | Spatial grid |
| **Convolution** | 1D (time axis) | 2D (height × width) |
| **Kernel** | $\mathbb{R}^{K}$ | $\mathbb{R}^{F_H \times F_W \times C}$ |
| **Typical input size** | ~16K samples | ~27K pixels (96×96×3) |
| **MACs per layer** | ~0.4M | ~40M |
| **Relative complexity** | 1× | ~100× |
| **Memory bandwidth** | ~16 KB/sec | ~270 KB/sec |

### Why 2D Convolution?

1. **Preserves spatial relationships**: "head above shoulders"
2. **Detects 2D patterns**: Edges, corners, shapes
3. **Enables hierarchical features**: Low-level edges → high-level objects
4. **Natural fit for grid data**: Images, not sequences

### Why Person Detection is Harder Than Speech

1. **Higher dimensionality**: 2D vs 1D
2. **More compute**: ~100× operations per layer
3. **More memory**: ~17× bandwidth requirement
4. **Richer features**: Spatial hierarchy at every level

**Therefore**: TinyML person detection requires aggressive optimization (quantization, small models, low resolution) to fit on Arduino Nano 33 Sense with 256 KB RAM and 1 MB Flash.

---

## Mathematical Notation Reference

| Symbol | Meaning |
|--------|---------|
| $\mathbb{R}$ | Set of real numbers |
| $\mathbb{R}^{n}$ | n-dimensional real vector space |
| $H, W$ | Height, Width (image dimensions) |
| $C$ | Number of channels (3 for RGB, 1 for grayscale) |
| $B$ | Batch size (number of images processed together) |
| $F_H, F_W$ | Filter height, width |
| $K$ | Kernel size (1D convolution) |
| $T$ | Time steps (audio samples) |
| $*$ | Convolution operator |
| $\sum$ | Summation |

---

**End of Mathematical Explanation**
