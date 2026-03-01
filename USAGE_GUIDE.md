# Complete TinyML Person Detection System - Usage Guide

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Data Collection](#data-collection)
4. [Training Pipeline](#training-pipeline)
5. [Arduino Deployment](#arduino-deployment)
6. [Performance Optimization](#performance-optimization)
7. [Troubleshooting](#troubleshooting)

---

## System Requirements

### For Training (GPU Machine)
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows 10+
- **Python**: 3.9 or 3.10
- **GPU**: NVIDIA GPU with 4GB+ VRAM (optional but recommended)
- **RAM**: 8GB+ system RAM
- **Storage**: 10GB+ free space

### For Deployment (Arduino)
- **Board**: Arduino Nano 33 BLE Sense
- **Camera**: ArduCam OV7675
- **Power**: 3.3V ⚠️ (NOT 5V!)

---

## Installation

### Step 1: Set Up Python Environment

```bash
# Create virtual environment (recommended)
python3.9 -m venv tinyml_env
source tinyml_env/bin/activate  # On Windows: tinyml_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
```

### Step 2: Install Arduino IDE

1. Download Arduino IDE 2.0+ from https://www.arduino.cc/en/software
2. Install for your OS
3. Install required libraries:
   - Open Arduino IDE
   - Go to Tools → Manage Libraries
   - Search and install:
     - "Arduino_TensorFlowLite" by TensorFlow Authors
     - "ArduCAM" by Lee

### Step 3: Clone or Download Project

```bash
cd ~/Documents/dynamic/face\ detectors/
ls  # Verify all files are present
```

Expected files:
```
├── DATASET_DESIGN.md
├── MATHEMATICAL_EXPLANATION.md
├── ARDUINO_DEPLOYMENT_GUIDE.md
├── README.md
├── requirements.txt
├── preprocess_dataset.py
├── model_architecture.py
├── train_model.py
├── convert_to_tflite.py
├── performance_metrics.py
├── run_pipeline.sh
└── arduino_person_detection/
    └── arduino_person_detection.ino
```

---

## Data Collection

### Minimum Dataset

Create this folder structure:
```
dataset/
├── person/
│   ├── img_0001.jpg
│   ├── img_0002.jpg
│   └── ... (1,000+ images)
└── no_person/
    ├── img_0001.jpg
    ├── img_0002.jpg
    └── ... (1,000+ images)
```

### Collection Guidelines

#### Person Class
- **Variety**: Different people, ages, clothing
- **Distances**: Close (1-2m), medium (2-5m), far (5-10m)
- **Orientations**: Frontal, profile, oblique, back
- **Lighting**: Indoor, outdoor, bright, dim
- **Contexts**: Home, office, street, park

#### No Person Class
- **Indoor**: Rooms, furniture, walls, doors
- **Outdoor**: Landscapes, buildings, streets, trees
- **Objects**: Cars, bikes, animals, plants
- **Hard negatives**: Statues, mannequins, posters

### Quality Checklist
- ✓ No blur (sharp focus)
- ✓ Proper exposure (not too dark/bright)
- ✓ No file corruption
- ✓ Diverse backgrounds
- ✓ Balanced class distribution

See [DATASET_DESIGN.md](DATASET_DESIGN.md) for detailed strategy.

---

## Training Pipeline

### Option 1: Automated Pipeline (Recommended)

Run the complete pipeline with one command:

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

This will:
1. Preprocess dataset
2. Train model
3. Convert to TFLite
4. Prepare Arduino deployment

### Option 2: Manual Step-by-Step

#### Step 1: Preprocess Data

```bash
python preprocess_dataset.py \
  --input_dir dataset/ \
  --output_dir preprocessed_dataset/ \
  --size 96 \
  --verify
```

**Options**:
- `--size 96`: Image size (96x96 default, try 64 for faster inference)
- `--grayscale`: Convert to grayscale (saves 3× memory)
- `--no-augmentation`: Disable augmentation

**Output**: `preprocessed_dataset/` with:
- `train/` (70% of data)
- `val/` (15% of data)
- `test/` (15% of data)
- `metadata.json` (configuration)

#### Step 2: Train Model

```bash
python train_model.py \
  --dataset_dir preprocessed_dataset/ \
  --model_size auto \
  --batch_size 32 \
  --epochs 50 \
  --patience 10 \
  --output_dir training_output/
```

**Options**:
- `--model_size auto`: Auto-select architecture (recommended)
- `--model_size tiny`: 30K params (~30 KB)
- `--model_size small`: 50K params (~50 KB)
- `--model_size medium`: 80K params (~80 KB)
- `--batch_size 32`: Adjust based on GPU memory
- `--epochs 50`: Max epochs (early stopping enabled)
- `--patience 10`: Stop if no improvement for 10 epochs

**Duration**: 10-30 minutes on GPU, 1-3 hours on CPU

**Output**: `training_output/` with:
- `best_model.h5`: Best model checkpoint
- `training_history.png`: Training curves
- `training_summary.json`: Metrics summary
- `evaluation_results.json`: Test set results

#### Step 3: Convert to TensorFlow Lite

```bash
python convert_to_tflite.py \
  --model training_output/best_model.h5 \
  --dataset_dir preprocessed_dataset/ \
  --output_dir tflite_models/
```

**Output**: `tflite_models/` with:
- `person_detector_int8.tflite`: Quantized model
- `person_detector_float32.tflite`: Float model (for comparison)
- `model_data.h`: C header for Arduino
- `conversion_summary.json`: Size and accuracy report

**Check**: Model size should be ≤100 KB. If larger:
- Use smaller architecture (`--model_size tiny`)
- Reduce input resolution (`--size 64`)
- Switch to grayscale

---

## Arduino Deployment

### Step 1: Hardware Setup

Wire ArduCam to Arduino Nano 33 Sense:

```
ArduCam OV7675  →  Arduino Nano 33 Sense
CS              →  D7
MOSI            →  D11 (SPI MOSI)
MISO            →  D12 (SPI MISO)
SCK             →  D13 (SPI SCK)
SDA             →  A4  (I2C SDA)
SCL             →  A5  (I2C SCL)
VCC             →  3.3V  ⚠️ NOT 5V!
GND             →  GND
```

**Critical**: ArduCam operates at 3.3V. Do NOT use 5V or it will be damaged.

### Step 2: Prepare Arduino Sketch

1. Ensure `model_data.h` is in `arduino_person_detection/` folder:
   ```bash
   ls arduino_person_detection/
   # Should show: arduino_person_detection.ino, model_data.h
   ```

2. If missing, copy from `tflite_models/`:
   ```bash
   cp tflite_models/model_data.h arduino_person_detection/
   ```

### Step 3: Configure Arduino IDE

1. Open Arduino IDE
2. Go to **Tools → Board → Arduino Mbed OS Nano Boards → Arduino Nano 33 BLE**
3. Connect Arduino via USB
4. Go to **Tools → Port** and select your Arduino's port

### Step 4: Adjust Memory Settings (If Needed)

Edit `arduino_person_detection.ino` if you encounter memory issues:

```cpp
// Reduce tensor arena if model is smaller
constexpr int kTensorArenaSize = 100 * 1024;  // Try 100 KB instead of 120 KB

// Switch to grayscale if using grayscale model
#define IMAGE_CHANNELS 1  // Change from 3 to 1

// Reduce input size if using 64x64 model
#define IMAGE_WIDTH 64   // Change from 96
#define IMAGE_HEIGHT 64  // Change from 96
```

**Important**: These settings must match your trained model's input shape.

### Step 5: Upload Firmware

1. Click **Verify** button (✓) to compile
2. Wait for compilation (2-3 minutes)
3. Check for errors in console
4. Click **Upload** button (→)
5. Wait for upload (30-60 seconds)
6. Arduino will reset automatically

### Step 6: Monitor Output

1. Open **Tools → Serial Monitor**
2. Set baud rate to **115200**
3. You should see initialization output:

```
====================================
TinyML Person Detection
Arduino Nano 33 Sense + ArduCam
====================================

[1/5] Initializing ArduCam...
✓ Camera initialized

[2/5] Setting up TensorFlow Lite...
✓ TensorFlow Lite ready

[3/5] Model Information:
  Model size: 52341 bytes (51.11 KB)
  Input shape: 96 x 96 x 3
  Input type: INT8
  Output classes: 2

[4/5] Memory Usage:
  Tensor arena: 120 KB
  Arena used: 87 KB
  Image buffer: 27.00 KB
  Total RAM usage: 147 KB
  ✓ RAM usage within limits

[5/5] Running test inference...
✓ Test inference successful (142 ms)

====================================
System ready! Starting detection...
====================================

[Frame 1] person (87.3%) | Inference: 143 ms | FPS: 7.0
[Frame 2] no_person (92.1%) | Inference: 145 ms | FPS: 6.9
```

---

## Performance Optimization

### Improve Accuracy

1. **Collect more diverse data**
   - Add challenging examples
   - Include edge cases (dim lighting, occlusions)
   - Balance class distribution

2. **Increase model capacity**
   - Use `medium` instead of `small`
   - Train longer (more epochs)

3. **Better preprocessing**
   - Ensure consistent normalization
   - Check data augmentation

### Improve Speed (FPS)

1. **Reduce input resolution**
   - 96×96 → 64×64 (2.25× faster)

2. **Use grayscale**
   - RGB → Grayscale (1.5-2× faster)

3. **Smaller model**
   - `medium` → `small` → `tiny`

4. **Optimize Arduino code**
   - Disable Serial prints in production
   - Use MicroMutableOpResolver instead of AllOpsResolver

### Reduce Memory Usage

1. **Smaller tensor arena**
   ```cpp
   constexpr int kTensorArenaSize = 100 * 1024;  // Instead of 120 KB
   ```

2. **Grayscale input**
   - 27 KB → 9 KB image buffer

3. **Lower resolution**
   - 96×96 → 64×64

---

## Troubleshooting

### Training Issues

**Problem**: ImportError: No module named 'tensorflow'  
**Solution**:
```bash
pip install tensorflow==2.15.0
```

**Problem**: CUDA out of memory  
**Solution**:
```bash
# Reduce batch size
python train_model.py --batch_size 16  # Instead of 32
```

**Problem**: Low accuracy (<60%)  
**Solution**:
- Collect more diverse data (especially hard negatives)
- Train longer (increase epochs)
- Try larger model architecture
- Check for mislabeled images

### Arduino Issues

**Problem**: 'model_data.h: No such file or directory'  
**Solution**:
```bash
cp tflite_models/model_data.h arduino_person_detection/
```

**Problem**: region 'FLASH' overflowed  
**Solution**:
- Model is too large
- Use smaller architecture or reduce resolution
- Ensure INT8 quantization was applied

**Problem**: region 'RAM' overflowed  
**Solution**:
```cpp
// Reduce tensor arena in .ino file
constexpr int kTensorArenaSize = 80 * 1024;  // Try 80 KB
```

**Problem**: Camera not detected  
**Solution**:
- Check wiring (especially CS pin = D7)
- Verify 3.3V power supply
- Test I2C communication with I2C scanner

**Problem**: Low FPS (<5)  
**Solution**:
- Reduce input resolution (96 → 64)
- Switch to grayscale
- Use smaller model
- Disable Serial debugging

**Problem**: Random predictions  
**Solution**:
- Check preprocessing matches training
- Verify model was quantized correctly
- Test with known images first

---

## Advanced Topics

### Custom Model Architecture

Edit `model_architecture.py` to create custom architectures:

```python
def build_custom_model(self):
    model = models.Sequential([
        layers.Input(shape=self.input_shape),
        # Your custom layers here
        layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        # ... more layers ...
        layers.Dense(self.num_classes, activation='softmax')
    ])
    return model
```

### Transfer Learning

Use a pre-trained model as starting point:

```python
from tensorflow.keras.applications import MobileNetV3Small

base_model = MobileNetV3Small(
    include_top=False,
    input_shape=(96, 96, 3),
    weights='imagenet'
)

# Freeze base layers
base_model.trainable = False

# Add custom head
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(2, activation='softmax')
])
```

**Note**: May exceed size constraints. Use post-training quantization and pruning.

### Real-Time Metrics on Arduino

Collect metrics during deployment:

```cpp
// In Arduino code, add to loop():
if (frame_count % 100 == 0) {
  float avg_latency = total_inference_time / frame_count;
  float accuracy = correct_predictions / (float)frame_count;
  
  Serial.print("Avg latency: ");
  Serial.print(avg_latency);
  Serial.print(" ms | Accuracy: ");
  Serial.println(accuracy * 100, 2);
}
```

---

## Performance Benchmarks

### Training Time (on GPU)
- **Small model (50K params)**: 10-15 minutes
- **Medium model (80K params)**: 15-25 minutes
- **With 4,000 training images**

### Inference Speed (Arduino Nano 33 Sense)
| Configuration | Latency | FPS | Accuracy |
|---------------|---------|-----|----------|
| 96×96 RGB | 140-150 ms | 6-7 | 80-85% |
| 96×96 Grayscale | 100-120 ms | 8-10 | 75-80% |
| 64×64 Grayscale | 60-80 ms | 12-16 | 70-75% |

### Memory Usage
| Configuration | Model Size | RAM Usage |
|---------------|------------|-----------|
| Medium (96×96 RGB) | 75-85 KB | 170-190 KB |
| Small (96×96 Gray) | 45-55 KB | 130-150 KB |
| Tiny (64×64 Gray) | 25-35 KB | 100-120 KB |

---

## Next Steps

After successful deployment:

1. **Optimize for your use case**
   - Tune confidence threshold
   - Adjust inference frequency
   - Add debouncing logic

2. **Add functionality**
   - Trigger LED/buzzer on detection
   - Send BLE notifications
   - Log events to SD card
   - Connect to IoT platform

3. **Power optimization**
   - Implement sleep modes
   - Reduce inference frequency
   - Use battery power

4. **Expand capabilities**
   - Multi-class detection (person, car, animal)
   - Object tracking across frames
   - Gesture recognition

---

## Resources

- **TensorFlow Lite Micro**: https://www.tensorflow.org/lite/microcontrollers
- **Arduino Nano 33 BLE Sense**: https://docs.arduino.cc/hardware/nano-33-ble-sense
- **ArduCam Documentation**: http://www.arducam.com/downloads/
- **TinyML Book**: "TinyML" by Pete Warden & Daniel Situnayake

---

**Good luck with your TinyML project! 🚀**
