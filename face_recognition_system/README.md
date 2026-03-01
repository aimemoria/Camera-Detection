# TinyML Face Recognition System

## Overview

A complete face recognition system for Arduino Nano 33 BLE Sense Rev2 with ArduCam Mini 2MP OV2640. Designed to meet CST-440 assignment requirements:

- ✅ Detect at least 5 different persons
- ✅ Two-stage pipeline (Human Detection → Identity Recognition)
- ✅ Unknown person detection via confidence thresholding
- ✅ Written (Serial) AND auditory (buzzer) feedback
- ✅ Target: 99% accuracy (minimum: 80%)

## System Architecture

```
Image → Stage A (Person?) → No  → "No Person" + low beep
                          ↓
                         Yes
                          ↓
        Stage B (Who?) → confidence < 70% → "Unknown" + warble
                       ↓
                    confidence ≥ 70%
                       ↓
                    "Person X" + unique beep pattern
```

## Project Structure

```
face_recognition_system/
├── A_HARDWARE_WIRING.md          # Hardware setup guide
├── B_DATA_COLLECTION_PROTOCOL.md # How to collect training data
├── C_preprocess_and_augment.py   # Data preprocessing pipeline
├── D_model_architecture.py       # Neural network definitions
├── E_train_model.py              # Training script
├── F_quantize_model.py           # INT8 quantization
├── G_arduino_firmware/           # Arduino sketch folder
│   ├── G_arduino_firmware.ino    # Main firmware
│   ├── stage_a_model.h           # Stage A model (placeholder)
│   └── stage_b_model.h           # Stage B model (placeholder)
├── H_TESTING_PROTOCOL.md         # Trial test documentation
├── I_SCIENTIFIC_REPORT_OUTLINE.md # CST-440 report template
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Quick Start

### 1. Setup Environment

```bash
cd face_recognition_system
python -m venv venv
source venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt
```

### 2. Wire Hardware

Follow [A_HARDWARE_WIRING.md](A_HARDWARE_WIRING.md):
- ArduCam CS → D7
- ArduCam MOSI → D11
- ArduCam MISO → D12
- ArduCam SCK → D13
- ArduCam SDA → A4
- ArduCam SCL → A5
- ArduCam VCC → 3.3V
- Buzzer → D6 (optional)

### 3. Collect Data

Follow [B_DATA_COLLECTION_PROTOCOL.md](B_DATA_COLLECTION_PROTOCOL.md):

```
dataset/
├── stage_a/
│   ├── person/         # 500+ images (from all 5 persons)
│   └── no_person/      # 500+ images (empty backgrounds)
├── stage_b/
│   ├── person1/        # 100+ images
│   ├── person2/        # 100+ images
│   ├── person3/        # 100+ images
│   ├── person4/        # 100+ images
│   └── person5/        # 100+ images
└── unknown_test/
    ├── other_person1/  # 50+ images (NOT in training)
    └── other_person2/  # 50+ images
```

### 4. Preprocess Data

```bash
python C_preprocess_and_augment.py \
    --dataset_dir dataset \
    --output_dir processed \
    --augment_train \
    --augmentations 3
```

### 5. Train Models

```bash
python E_train_model.py \
    --data_dir processed \
    --output_dir models \
    --epochs 100
```

### 6. Quantize Models

```bash
python F_quantize_model.py \
    --model_dir models \
    --data_dir processed \
    --output_dir tflite \
    --validate
```

### 7. Deploy to Arduino

1. Copy generated headers to Arduino sketch:
   ```bash
   cp tflite/stage_a_model.h G_arduino_firmware/
   cp tflite/stage_b_model.h G_arduino_firmware/
   ```

2. Open `G_arduino_firmware/G_arduino_firmware.ino` in Arduino IDE

3. Install required libraries:
   - ArduCAM
   - TensorFlowLite (Arduino_TensorFlowLite)

4. Select board: Arduino Nano 33 BLE

5. Upload firmware

### 8. Test System

Follow [H_TESTING_PROTOCOL.md](H_TESTING_PROTOCOL.md):
- Open Serial Monitor (115200 baud)
- Press Enter or send 'c' to trigger capture
- Record results in trial table

## Hardware Requirements

| Component | Model | Notes |
|-----------|-------|-------|
| Microcontroller | Arduino Nano 33 BLE Sense Rev2 | 256KB RAM, 1MB Flash |
| Camera | ArduCam Mini 2MP OV2640 | With 8MB FIFO |
| Audio | Passive Piezo Buzzer | Optional, on D6 |

## Model Specifications

| Model | Parameters | INT8 Size | Input | Output |
|-------|------------|-----------|-------|--------|
| Stage A | ~15,000 | ~15-20 KB | 96×96×1 | 2 classes |
| Stage B | ~25,000 | ~25-30 KB | 96×96×1 | 5 classes |

## Memory Budget

| Component | Size |
|-----------|------|
| Stage A Model | ~20 KB |
| Stage B Model | ~30 KB |
| Tensor Arena | ~100 KB |
| Image Buffer | ~9 KB |
| **Total RAM** | ~159 KB / 256 KB |

## Audio Feedback

| Result | Pattern |
|--------|---------|
| No Person | 1 low beep (200Hz) |
| Person 1 | 2 high beeps (1000Hz) |
| Person 2 | 1 long mid beep (800Hz) |
| Person 3 | 3 ascending beeps |
| Person 4 | 2 descending beeps |
| Person 5 | Short + long |
| Unknown | Low warble |

## Troubleshooting

### Camera Not Detected
- Check SPI wiring (CS, MOSI, MISO, SCK)
- Verify 3.3V power (NOT 5V!)
- Run I2C scanner test from A_HARDWARE_WIRING.md

### Low Accuracy
- Increase training data variety (angles, lighting)
- Adjust unknown threshold (default: 0.70)
- Ensure test conditions match training conditions

### Memory Issues
- Use ultra-tiny model variants (see D_model_architecture.py)
- Reduce tensor arena size
- Reduce image resolution to 64×64

## License

MIT License - See LICENSE file

## References

- TensorFlow Lite for Microcontrollers
- ArduCAM Documentation
- Arduino Nano 33 BLE Sense Rev2 Datasheet
