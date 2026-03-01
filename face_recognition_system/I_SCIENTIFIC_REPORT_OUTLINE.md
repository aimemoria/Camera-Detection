# I. Scientific Report Outline (CST-440 Format)

## TinyML Face Recognition System for Arduino Nano 33 BLE Sense

---

## Title Page

**Title**: TinyML-Based Face Recognition System Using Two-Stage Neural Network Architecture on Arduino Nano 33 BLE Sense Rev2

**Authors**: [Student Name(s)]

**Course**: CST-440: Embedded Machine Learning

**Institution**: [University Name]

**Date**: [Submission Date]

---

## Abstract

[150-300 words summarizing the entire project]

This paper presents the design, implementation, and evaluation of a TinyML-based face recognition system deployed on the Arduino Nano 33 BLE Sense Rev2 microcontroller. The system employs a two-stage neural network architecture: Stage A performs binary person detection (person vs. no person), and Stage B performs multi-class face recognition among five enrolled individuals. Unknown persons are identified through confidence thresholding rather than explicit classification.

The models were trained using TensorFlow/Keras and quantized to INT8 format using TensorFlow Lite for Microcontrollers (TFLM), achieving combined model sizes of approximately XX KB. The system captures images using an ArduCam Mini 2MP OV2640 camera module with 8MB FIFO buffer, preprocesses them to 96×96 grayscale, and performs on-device inference.

Experimental results demonstrate XX% accuracy on the test dataset, with an average inference time of XXX milliseconds. The system provides both serial text output and auditory feedback through distinct buzzer patterns for each recognized individual. The achieved accuracy exceeds the 80% threshold required by the project rubric, validating the feasibility of face recognition on resource-constrained microcontrollers.

**Keywords**: TinyML, Face Recognition, Edge AI, Neural Networks, Quantization, Arduino, Embedded Systems

---

## 1. Introduction

### 1.1 Background and Motivation

[Discuss the importance of edge AI and face recognition]
- Growth of IoT and edge computing
- Privacy benefits of on-device inference
- Energy efficiency of TinyML solutions
- Applications: access control, personalization, security

### 1.2 Problem Statement

[Define the specific problem being solved]
- Requirement to recognize 5+ specific individuals
- Constraint of microcontroller resources (256KB RAM, 1MB Flash)
- Need for both human detection and identity verification
- Unknown person detection capability

### 1.3 Objectives

1. Design a two-stage neural network architecture suitable for TinyML deployment
2. Implement face detection (person vs. no person) as Stage A
3. Implement face recognition (identify specific individuals) as Stage B
4. Achieve at least 80% recognition accuracy (target: 99%)
5. Deploy on Arduino Nano 33 BLE Sense Rev2 with ArduCam
6. Provide written (Serial) and auditory (buzzer) feedback

### 1.4 Scope and Limitations

- Recognition limited to 5 enrolled individuals
- Unknown persons detected via confidence thresholding
- Grayscale images at 96×96 resolution
- Controlled indoor environment

### 1.5 Report Organization

- Section 2: Literature review of TinyML and face recognition
- Section 3: System design and methodology
- Section 4: Implementation details
- Section 5: Experimental results and evaluation
- Section 6: Discussion and analysis
- Section 7: Conclusions and future work

---

## 2. Literature Review

### 2.1 TinyML and Edge Computing

[Review of TinyML literature]
- Definition and characteristics of TinyML
- Comparison with cloud-based and mobile ML
- TensorFlow Lite for Microcontrollers (TFLM)
- Model optimization techniques (quantization, pruning)

### 2.2 Face Recognition Systems

[Overview of face recognition approaches]
- Traditional methods (Eigenfaces, Fisherfaces)
- Deep learning approaches (CNNs, FaceNet, ArcFace)
- Compact architectures (MobileNet, ShuffleNet)
- Two-stage detection-recognition pipelines

### 2.3 Embedded Face Recognition

[Review of embedded/TinyML face recognition work]
- Previous work on microcontroller-based face recognition
- Trade-offs between accuracy and model size
- Challenges of limited memory and compute

### 2.4 Related Systems and Approaches

[Comparison with similar projects]
- Person detection for keyword spotting (as baseline)
- Wake-word detection analogy
- Multi-stage inference pipelines

---

## 3. System Design and Methodology

### 3.1 System Architecture

[High-level system design]

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│  ArduCam    │───>│  Preprocess  │───>│   Stage A    │
│  OV2640     │    │  96x96 Gray  │    │  Person Det  │
└─────────────┘    └──────────────┘    └──────┬───────┘
                                              │
                        ┌─────────────────────┘
                        │ Person Detected?
                        ▼
                  ┌─────┴─────┐
                  │  Stage B  │
                  │  Face ID  │
                  └─────┬─────┘
                        │
                        ▼
                  ┌───────────┐
                  │ Threshold │──> Unknown / Person 1-5
                  └───────────┘
```

### 3.2 Hardware Platform

[Description of hardware components]

| Component | Specification |
|-----------|---------------|
| Microcontroller | Arduino Nano 33 BLE Sense Rev2 |
| Processor | nRF52840 (ARM Cortex-M4 @ 64MHz) |
| RAM | 256 KB |
| Flash | 1 MB |
| Camera | ArduCam Mini 2MP OV2640 |
| Camera FIFO | 8MB (W25Q64JV) |
| Audio Output | Passive Piezo Buzzer |

### 3.3 Neural Network Design

#### 3.3.1 Stage A: Person Detection Model

[Architecture details]
- Input: 96×96×1 grayscale image
- Architecture: 4-layer CNN (Conv→BN→ReLU blocks)
- Output: 2 classes (no_person, person)
- Parameters: ~15,000
- Quantized size: ~15-20 KB

| Layer | Output Shape | Parameters |
|-------|--------------|------------|
| Conv2D (8, 3×3, s=2) | 48×48×8 | XXX |
| Conv2D (16, 3×3, s=2) | 24×24×16 | XXX |
| Conv2D (24, 3×3, s=2) | 12×12×24 | XXX |
| Conv2D (32, 3×3, s=2) | 6×6×32 | XXX |
| GlobalAvgPool | 32 | 0 |
| Dense (2) | 2 | XXX |

#### 3.3.2 Stage B: Face Recognition Model

[Architecture details]
- Input: 96×96×1 grayscale image
- Architecture: MobileNet-style depthwise separable convolutions
- Output: 5 classes (person1, person2, person3, person4, person5)
- Parameters: ~25,000
- Quantized size: ~25-30 KB

| Layer | Output Shape | Parameters |
|-------|--------------|------------|
| Conv2D (16, 3×3, s=2) | 48×48×16 | XXX |
| DWConv + PWConv (32) | 24×24×32 | XXX |
| DWConv + PWConv (48) | 12×12×48 | XXX |
| DWConv + PWConv (64) | 6×6×64 | XXX |
| GlobalAvgPool | 64 | 0 |
| Dense (32, relu) | 32 | XXX |
| Dense (5, softmax) | 5 | XXX |

### 3.4 Unknown Detection Strategy

[Explanation of confidence thresholding]
- Unknown persons are NOT a separate class
- Detection via confidence threshold on Stage B output
- If max(softmax) < threshold → classify as "Unknown"
- Threshold tuned using held-out unknown test data

### 3.5 Data Collection Protocol

[Summary of data collection approach]
- 5 known persons: 100-200 images each
- Multiple orientations, lighting conditions, backgrounds
- Background (no_person): 500-1000 images
- Unknown test persons: 3+ individuals, 50 images each

---

## 4. Implementation

### 4.1 Dataset Preparation

[Details of data collection and preprocessing]
- Image capture using ArduCam or smartphone
- Preprocessing: resize to 96×96, convert to grayscale, normalize
- Data augmentation: rotation, brightness, contrast, flip
- Train/validation/test split: 70%/15%/15%

### 4.2 Model Training

[Training procedure details]
- Framework: TensorFlow/Keras
- Optimizer: Adam (lr=0.001)
- Loss: Sparse Categorical Crossentropy
- Batch size: 32
- Early stopping with patience=15
- Learning rate reduction on plateau

### 4.3 Model Quantization

[INT8 quantization process]
- TensorFlow Lite Converter
- Representative dataset for calibration (100 samples)
- Full integer quantization (INT8 weights and activations)
- C header generation for Arduino inclusion

### 4.4 Arduino Firmware

[Firmware implementation details]
- TensorFlow Lite Micro integration
- ArduCAM library for image capture
- Shared tensor arena (100 KB) for both models
- Sequential inference (Stage A → Stage B)
- Serial output and buzzer feedback

### 4.5 Memory Management

[Memory budget analysis]

| Component | Size | Notes |
|-----------|------|-------|
| Stage A Model | ~20 KB | INT8 quantized |
| Stage B Model | ~30 KB | INT8 quantized |
| Tensor Arena | ~100 KB | Shared between stages |
| Image Buffer | ~9 KB | 96×96×1 bytes |
| Stack/Variables | ~20 KB | Estimated |
| **Total RAM** | ~159 KB | of 256 KB available |

---

## 5. Experimental Results

### 5.1 Training Results

[Training performance]

#### Stage A Training Curves
[Include accuracy and loss plots]

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| Accuracy | XX% | XX% | XX% |
| Loss | X.XX | X.XX | X.XX |

#### Stage B Training Curves
[Include accuracy and loss plots]

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| Accuracy | XX% | XX% | XX% |
| Loss | X.XX | X.XX | X.XX |

### 5.2 Quantization Impact

[Effect of INT8 quantization]

| Model | Float32 Acc | INT8 Acc | Size Reduction |
|-------|-------------|----------|----------------|
| Stage A | XX% | XX% | X.Xx |
| Stage B | XX% | XX% | X.Xx |

### 5.3 Confusion Matrices

[Stage A confusion matrix]
```
              Predicted
              No Person | Person
Actual ───────────────────────────
No Person |     XX     |   XX
Person    |     XX     |   XX
```

[Stage B confusion matrix]
```
              Predicted
              P1 | P2 | P3 | P4 | P5
Actual ─────────────────────────────
Person 1 |   XX | XX | XX | XX | XX
Person 2 |   XX | XX | XX | XX | XX
Person 3 |   XX | XX | XX | XX | XX
Person 4 |   XX | XX | XX | XX | XX
Person 5 |   XX | XX | XX | XX | XX
```

### 5.4 Unknown Threshold Tuning

[Threshold selection results]

| Threshold | Known Acc | Unknown Rej | Combined |
|-----------|-----------|-------------|----------|
| 0.50 | XX% | XX% | XX% |
| 0.60 | XX% | XX% | XX% |
| 0.70 | XX% | XX% | XX% |
| 0.80 | XX% | XX% | XX% |

**Selected threshold**: 0.XX

### 5.5 Live Trial Results

[Results from 10 trial tests per rubric]

| Trial | Subject | Expected | Result | Confidence | Pass |
|-------|---------|----------|--------|------------|------|
| 1 | Person 1 | Person 1 | | XX% | |
| 2 | Person 2 | Person 2 | | XX% | |
| ... | ... | ... | ... | ... | ... |
| 10 | Edge Case | | | XX% | |

**Overall Accuracy**: XX/10 = XX%

### 5.6 Inference Performance

[Timing measurements]

| Stage | Avg Time | Min | Max |
|-------|----------|-----|-----|
| Image Capture | XXX ms | XX | XX |
| Preprocessing | XX ms | XX | XX |
| Stage A Inference | XX ms | XX | XX |
| Stage B Inference | XX ms | XX | XX |
| **Total** | XXX ms | XX | XX |

---

## 6. Discussion

### 6.1 Analysis of Results

[Interpretation of experimental results]
- Comparison to target accuracy (80% minimum, 99% target)
- Analysis of confusion matrices
- Performance under different conditions

### 6.2 Comparison with Literature

[How results compare to related work]
- Performance vs. model size trade-off
- Comparison with other TinyML face recognition systems

### 6.3 Limitations

[Discussion of system limitations]
- Limited to 5 enrolled persons
- Controlled lighting environment
- Grayscale imaging limitations
- Single face per frame assumption

### 6.4 Sources of Error

[Analysis of failure cases]
- Similar-looking individuals
- Extreme lighting variations
- Unusual poses or expressions
- Partial occlusion

---

## 7. Conclusion

### 7.1 Summary of Achievements

[Recap of what was accomplished]
- Successfully implemented two-stage face recognition
- Achieved XX% accuracy (meets/exceeds 80% requirement)
- Deployed on resource-constrained microcontroller
- Provided both text and audio feedback

### 7.2 Contributions

1. Two-stage TinyML architecture for face recognition
2. INT8 quantized models fitting within microcontroller constraints
3. Unknown detection via confidence thresholding
4. Complete end-to-end system from data collection to deployment

### 7.3 Future Work

[Suggestions for improvement]
- Increase to 10+ enrolled persons
- Add continual learning for new enrollments
- Implement face detection for region of interest
- Explore smaller model architectures (knowledge distillation)
- Add anti-spoofing (liveness detection)

---

## References

[IEEE or ACM format]

[1] A. Author, B. Author, "Title of Paper," *Journal Name*, vol. X, no. X, pp. XX-XX, Year.

[2] TensorFlow Lite for Microcontrollers. [Online]. Available: https://www.tensorflow.org/lite/microcontrollers

[3] ArduCAM. [Online]. Available: https://www.arducam.com/

[4] P. Warden and D. Situnayake, *TinyML: Machine Learning with TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers*. O'Reilly Media, 2019.

[5] ...

---

## Appendices

### Appendix A: Full Source Code Listings

[Reference to attached code files or GitHub repository]

### Appendix B: Hardware Wiring Diagram

[Include diagram from A_HARDWARE_WIRING.md]

### Appendix C: Complete Trial Logs

[Raw data from all test trials]

### Appendix D: Training Logs

[TensorBoard screenshots or raw training output]

---

## Figures List

- Figure 1: System Architecture Block Diagram
- Figure 2: Hardware Setup Photo
- Figure 3: Stage A Model Architecture
- Figure 4: Stage B Model Architecture
- Figure 5: Stage A Training Curves
- Figure 6: Stage B Training Curves
- Figure 7: Stage A Confusion Matrix
- Figure 8: Stage B Confusion Matrix
- Figure 9: Unknown Threshold Tuning Curves
- Figure 10: Sample Recognition Results

---

## Tables List

- Table 1: Hardware Specifications
- Table 2: Stage A Layer Details
- Table 3: Stage B Layer Details
- Table 4: Memory Budget
- Table 5: Training Results Summary
- Table 6: Quantization Impact
- Table 7: Live Trial Results
- Table 8: Inference Timing

---

## Checklist Before Submission

- [ ] Abstract is 150-300 words
- [ ] All sections completed
- [ ] Figures and tables numbered and captioned
- [ ] References properly formatted
- [ ] Spelling and grammar checked
- [ ] Page numbers included
- [ ] Code attached/linked
- [ ] Results clearly state accuracy achieved vs. 80% target
