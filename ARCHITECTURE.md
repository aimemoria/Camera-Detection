# TinyML Person Detection - System Architecture

## Overall System Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          COMPLETE PIPELINE                                  │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  1. DATA         │      │  2. TRAINING     │      │  3. DEPLOYMENT   │
│  COLLECTION      │─────▶│  (GPU MACHINE)   │─────▶│  (ARDUINO)       │
│                  │      │                  │      │                  │
│  - Camera        │      │  - Python        │      │  - Arduino IDE   │
│  - Smartphone    │      │  - TensorFlow    │      │  - C++           │
│  - 2,000+ images │      │  - GPU (4GB+)    │      │  - 256KB RAM     │
└──────────────────┘      └──────────────────┘      └──────────────────┘
```

---

## Detailed Component Architecture

### 1. Data Collection & Preprocessing

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA PREPARATION PIPELINE                        │
└─────────────────────────────────────────────────────────────────────┘

Raw Images                    Preprocessing                Output
─────────                     ─────────────                ──────

dataset/                                              preprocessed_dataset/
├─ person/          ┌───────────────────┐             ├─ train/
│  ├─ img1.jpg      │  1. Load Image    │             │  ├─ person/
│  ├─ img2.jpg  ───▶│  2. Resize        │───▶         │  │  └─ *.npy
│  └─ ...           │  3. Normalize     │             │  └─ no_person/
└─ no_person/       │  4. Augment       │             │     └─ *.npy
   ├─ img1.jpg      │  5. Save as .npy  │             ├─ val/
   ├─ img2.jpg      └───────────────────┘             │  └─ ... (same)
   └─ ...                                             └─ test/
                                                         └─ ... (same)

Operations:
• Resize: 96x96 or 64x64
• Normalize: [-1, 1] range
• Augment: Flip, brightness, rotation, noise
• Format: NumPy arrays (.npy)
```

### 2. Model Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                   CNN ARCHITECTURE (SMALL MODEL)                    │
└─────────────────────────────────────────────────────────────────────┘

Input: (1, 96, 96, 1)  [Batch, Height, Width, Channels]
         │
         ▼
┌──────────────────────────────────┐
│ Conv2D(16 filters, 3x3)          │  Output: (1, 96, 96, 16)
│ + BatchNormalization             │  Params: 448
│ + ReLU                           │
└──────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│ MaxPooling2D(2x2)                │  Output: (1, 48, 48, 16)
└──────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│ Conv2D(32 filters, 3x3)          │  Output: (1, 48, 48, 32)
│ + BatchNormalization             │  Params: 4,640
│ + ReLU                           │
└──────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│ MaxPooling2D(2x2)                │  Output: (1, 24, 24, 32)
└──────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│ Conv2D(32 filters, 3x3)          │  Output: (1, 24, 24, 32)
│ + ReLU                           │  Params: 9,248
└──────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│ MaxPooling2D(2x2)                │  Output: (1, 12, 12, 32)
└──────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│ GlobalAveragePooling2D           │  Output: (1, 32)
└──────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│ Dense(64)                        │  Output: (1, 64)
│ + ReLU                           │  Params: 2,112
│ + Dropout(0.4)                   │
└──────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│ Dense(2)                         │  Output: (1, 2)
│ + Softmax                        │  Params: 130
└──────────────────────────────────┘
         │
         ▼
    Prediction
  [no_person, person]

TOTAL PARAMETERS: ~16,500
QUANTIZED SIZE: ~17 KB (INT8)
```

### 3. Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING WORKFLOW                            │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────┐
│ Load Data       │
│ - Train (70%)   │
│ - Val (15%)     │
│ - Test (15%)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Build Model     │
│ - Auto-select   │
│ - Compile       │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ Training Loop                       │
│ ┌─────────────────────────────────┐ │
│ │ Epoch 1:                        │ │
│ │  - Forward pass                 │ │
│ │  - Compute loss                 │ │
│ │  - Backpropagation              │ │
│ │  - Update weights               │ │
│ │  - Validate                     │ │
│ └─────────────────────────────────┘ │
│ ┌─────────────────────────────────┐ │
│ │ Callbacks:                      │ │
│ │  - Early stopping (patience=10) │ │
│ │  - LR reduction (factor=0.5)    │ │
│ │  - Model checkpoint (best)      │ │
│ │  - TensorBoard logging          │ │
│ └─────────────────────────────────┘ │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ Evaluate        │
│ - Test accuracy │
│ - Per-class     │
│ - Confusion mtx │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Save Model      │
│ - best_model.h5 │
└─────────────────┘
```

### 4. Quantization & Conversion

```
┌─────────────────────────────────────────────────────────────────────┐
│                   TFLITE CONVERSION PIPELINE                        │
└─────────────────────────────────────────────────────────────────────┘

Keras Model (.h5)
    │
    │  Size: ~200 KB (float32)
    │
    ▼
┌──────────────────────────┐
│ TFLite Converter         │
│ - Load Keras model       │
│ - Convert graph          │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│ INT8 Quantization                    │
│ ┌──────────────────────────────────┐ │
│ │ 1. Representative Dataset        │ │
│ │    - 100 samples from train      │ │
│ │ 2. Calibration                   │ │
│ │    - Run inference               │ │
│ │    - Record min/max per layer    │ │
│ │ 3. Quantize Weights              │ │
│ │    - float32 → int8              │ │
│ │ 4. Set Input/Output to INT8      │ │
│ └──────────────────────────────────┘ │
└──────────┬───────────────────────────┘
           │
           ▼
    TFLite Model
 (.tflite, ~50 KB)
           │
           ▼
┌──────────────────────────┐
│ Validate                 │
│ - Run inference          │
│ - Check accuracy         │
│ - Compare with Keras     │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ Generate C Header        │
│ - Binary → hex array     │
│ - model_data.h           │
└──────────────────────────┘
```

### 5. Arduino Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ARDUINO SYSTEM ARCHITECTURE                      │
└─────────────────────────────────────────────────────────────────────┘

Hardware Layer
──────────────
┌──────────────┐          ┌──────────────┐
│  ArduCam     │  I2C/SPI │   Arduino    │
│  OV7675      │◀────────▶│   Nano 33    │
│              │          │   BLE Sense  │
│ - Image      │          │              │
│   capture    │          │ - 256KB RAM  │
│ - JPEG       │          │ - 1MB Flash  │
└──────────────┘          └──────────┬───┘
                                     │ USB
                                     │
                                     ▼
                              ┌──────────────┐
                              │   Computer   │
                              │ (Serial Mon) │
                              └──────────────┘

Software Architecture
────────────────────

┌─────────────────────────────────────────────────────────────────────┐
│                         FIRMWARE LAYERS                             │
├─────────────────────────────────────────────────────────────────────┤
│ Application Layer                                                   │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ • Main Loop                                                     │ │
│ │ • Capture → Preprocess → Infer → Output                        │ │
│ │ • Performance monitoring (FPS, latency)                         │ │
│ └─────────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│ Inference Layer (TensorFlow Lite Micro)                             │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ • MicroInterpreter                                              │ │
│ │ • Tensor Arena (120 KB)                                         │ │
│ │ • OpResolver (Conv2D, Pool, Dense, etc.)                        │ │
│ │ • Quantization/Dequantization                                   │ │
│ └─────────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│ Hardware Abstraction Layer                                          │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ • ArduCAM library (camera interface)                            │ │
│ │ • SPI (image data transfer)                                     │ │
│ │ • I2C (camera control)                                          │ │
│ │ • Serial (debug output)                                         │ │
│ └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘

Memory Layout
─────────────

┌────────────────────────────┐  ◀─ 0x00000000
│  Flash (1 MB)              │
│  ┌──────────────────────┐  │
│  │ Firmware Code        │  │  ~50-80 KB
│  ├──────────────────────┤  │
│  │ TFLite Model         │  │  ~20-80 KB
│  │ (model_data[])       │  │
│  ├──────────────────────┤  │
│  │ Arduino Libraries    │  │  ~30-50 KB
│  └──────────────────────┘  │
└────────────────────────────┘

┌────────────────────────────┐  ◀─ 0x20000000
│  RAM (256 KB)              │
│  ┌──────────────────────┐  │
│  │ Stack                │  │  ~20 KB
│  ├──────────────────────┤  │
│  │ Heap                 │  │  ~20 KB
│  ├──────────────────────┤  │
│  │ Tensor Arena         │  │  ~120 KB
│  │ (inference workspace)│  │
│  ├──────────────────────┤  │
│  │ Image Buffer         │  │  ~27 KB (96x96x3)
│  │ [H][W][C]            │  │  or 9 KB (grayscale)
│  ├──────────────────────┤  │
│  │ Global Variables     │  │  ~10 KB
│  └──────────────────────┘  │
│  Free: ~60 KB              │
└────────────────────────────┘
```

### 6. Inference Flow (Arduino)

```
┌─────────────────────────────────────────────────────────────────────┐
│                      REAL-TIME INFERENCE LOOP                       │
└─────────────────────────────────────────────────────────────────────┘

 ┌─────────────────────────────────────────────────────────────────┐
 │  Loop Start                                                     │
 └─────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  1. Capture Frame                                               │
 │  ┌───────────────────────────────────────────────────────────┐  │
 │  │  ArduCam.start_capture()                                  │  │
 │  │  Wait for CAP_DONE                                        │  │
 │  │  Read JPEG from FIFO                                      │  │
 │  │  Decode to RGB (or grayscale)                             │  │
 │  │  Store in image_buffer[96][96][3]                         │  │
 │  └───────────────────────────────────────────────────────────┘  │
 └─────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  2. Preprocess                                                  │
 │  ┌───────────────────────────────────────────────────────────┐  │
 │  │  for each pixel:                                          │  │
 │  │    normalized = (pixel / 127.5) - 1.0  # [-1, 1]         │  │
 │  │    if INT8:                                               │  │
 │  │      quantized = (normalized / scale) + zero_point       │  │
 │  │      input_tensor[i] = clip(quantized, -128, 127)        │  │
 │  └───────────────────────────────────────────────────────────┘  │
 └─────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  3. Run Inference                                               │
 │  ┌───────────────────────────────────────────────────────────┐  │
 │  │  start_time = millis()                                    │  │
 │  │  interpreter->Invoke()  ◀─── TFLite Micro Engine         │  │
 │  │  inference_time = millis() - start_time                   │  │
 │  └───────────────────────────────────────────────────────────┘  │
 └─────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  4. Get Prediction                                              │
 │  ┌───────────────────────────────────────────────────────────┐  │
 │  │  output_tensor = interpreter->output(0)                   │  │
 │  │  if INT8:                                                 │  │
 │  │    score = (output_int8 - zero_point) * scale            │  │
 │  │  no_person_score = output[0]                              │  │
 │  │  person_score = output[1]                                 │  │
 │  │  predicted_class = argmax(scores)                         │  │
 │  │  confidence = max(scores)                                 │  │
 │  └───────────────────────────────────────────────────────────┘  │
 └─────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  5. Output Results                                              │
 │  ┌───────────────────────────────────────────────────────────┐  │
 │  │  Serial.print("Frame X: ")                                │  │
 │  │  Serial.print(predicted_class == 1 ? "person" : "no")    │  │
 │  │  Serial.print(confidence * 100, "%")                      │  │
 │  │  Serial.print("Inference: ", inference_time, "ms")        │  │
 │  │  Serial.print("FPS: ", 1000/inference_time)               │  │
 │  └───────────────────────────────────────────────────────────┘  │
 └─────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  6. Update Metrics                                              │
 │  ┌───────────────────────────────────────────────────────────┐  │
 │  │  inference_count++                                        │  │
 │  │  total_inference_time += inference_time                   │  │
 │  │  frame_count++                                            │  │
 │  └───────────────────────────────────────────────────────────┘  │
 └─────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
                delay(100)  # Throttle
                   │
                   └──────────▶ Loop Start
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         END-TO-END DATA FLOW                        │
└─────────────────────────────────────────────────────────────────────┘

Real World          Training Phase                   Deployment Phase
───────────         ──────────────                   ────────────────

[Camera]                                             [ArduCam]
   │                                                     │
   │ Capture                                            │ Capture
   ▼                                                     ▼
[JPEG Images]                                        [96x96 Image]
   │                                                     │
   │ Collect                                            │ Preprocess
   ▼                                                     ▼
[dataset/]                                           [Normalized]
 ├─ person/                                             │
 └─ no_person/                                          │ Feed to model
   │                                                     ▼
   │ preprocess_dataset.py                      ┌──────────────┐
   ▼                                            │ TFLite Model │
[preprocessed_dataset/]                         │   (INT8)     │
 ├─ train/ (70%)                                └──────┬───────┘
 ├─ val/ (15%)                                         │
 └─ test/ (15%)                                        │ Inference
   │                                                    ▼
   │ train_model.py                              [Prediction]
   ▼                                              [0.87, 0.13]
[Keras Model]                                          │
 best_model.h5                                         │ Argmax
 (~200 KB, float32)                                    ▼
   │                                              "person" (87%)
   │ convert_to_tflite.py                              │
   ▼                                                    │ Output
[TFLite Model]                                         ▼
 person_detector_int8.tflite                     [Serial Monitor]
 (~50 KB, int8)                                   "Frame 1: person
   │                                               (87.3%) | 143ms"
   │ xxd -i
   ▼
[model_data.h]
 const unsigned char model_data[]
   │
   │ Upload to Arduino
   ▼
[Arduino Flash Memory]
```

---

## Performance Monitoring Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    METRICS COLLECTION SYSTEM                        │
└─────────────────────────────────────────────────────────────────────┘

Training Phase (Python)
──────────────────────

┌──────────────────────────────────────────────────────────────────┐
│ PerformanceMetrics Class                                         │
│ ┌──────────────────────────────────────────────────────────────┐ │
│ │ For each prediction:                                         │ │
│ │   metrics.update(y_true, y_pred, confidence, latency)        │ │
│ │                                                              │ │
│ │ Track:                                                       │ │
│ │   • True positives/negatives                                │ │
│ │   • False positives/negatives                               │ │
│ │   • Inference times (min/max/mean/p95/p99)                  │ │
│ │   • Predictions and ground truth                            │ │
│ └──────────────────────────────────────────────────────────────┘ │
│                                                                  │
│ Outputs:                                                         │
│   • Confusion matrix                                             │
│   • Per-class precision/recall/F1                                │
│   • Latency distribution                                         │
│   • Overall accuracy                                             │
│   • FPS statistics                                               │
└──────────────────────────────────────────────────────────────────┘

Deployment Phase (Arduino)
──────────────────────────

┌──────────────────────────────────────────────────────────────────┐
│ Real-Time Monitoring (Serial Output)                             │
│ ┌──────────────────────────────────────────────────────────────┐ │
│ │ Per-frame metrics:                                           │ │
│ │   Frame number                                               │ │
│ │   Predicted class                                            │ │
│ │   Confidence (%)                                             │ │
│ │   Inference time (ms)                                        │ │
│ │   FPS                                                        │ │
│ │                                                              │ │
│ │ Aggregate metrics (every 5 seconds):                         │ │
│ │   Total frames processed                                     │ │
│ │   Average inference time                                     │ │
│ │   Average FPS                                                │ │
│ │   Total inferences                                           │ │
│ └──────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TECHNOLOGY LAYERS                           │
└─────────────────────────────────────────────────────────────────────┘

Training Stack
─────────────

┌────────────────────────────────────────────────────────┐
│ Python 3.9/3.10                                        │
├────────────────────────────────────────────────────────┤
│ TensorFlow 2.15.0        │ Machine Learning Framework  │
│ NumPy 1.24.3             │ Numerical Computing         │
│ OpenCV 4.8.1             │ Image Processing            │
│ Matplotlib 3.8.0         │ Visualization               │
│ scikit-learn 1.3.2       │ Data Splitting, Metrics     │
└────────────────────────────────────────────────────────┘

Deployment Stack
───────────────

┌────────────────────────────────────────────────────────┐
│ Arduino C++                                            │
├────────────────────────────────────────────────────────┤
│ Arduino_TensorFlowLite   │ TFLite Micro Runtime        │
│ ArduCAM                  │ Camera Interface            │
│ Wire                     │ I2C Communication           │
│ SPI                      │ SPI Communication           │
└────────────────────────────────────────────────────────┘

Hardware
────────

┌────────────────────────────────────────────────────────┐
│ Arduino Nano 33 BLE Sense                              │
│ • nRF52840 MCU (ARM Cortex-M4)                         │
│ • 256 KB RAM, 1 MB Flash                               │
│ • 64 MHz clock                                         │
├────────────────────────────────────────────────────────┤
│ ArduCam OV7675                                         │
│ • VGA camera (640x480 max)                             │
│ • JPEG output                                          │
│ • I2C control, SPI data                                │
└────────────────────────────────────────────────────────┘
```

---

**Complete System Architecture Documentation**  
*For TinyML Person Detection on Arduino Nano 33 BLE Sense*
