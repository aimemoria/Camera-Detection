# Arduino Deployment Guide

## Prerequisites

### Hardware
- Arduino Nano 33 BLE Sense
- ArduCam OV7675 camera module
- USB cable (Micro-B or USB-C depending on board)
- Breadboard and jumper wires (optional)

### Software
- Arduino IDE 2.0+ (Download: https://www.arduino.cc/en/software)
- USB drivers for Arduino Nano 33 (usually auto-installed)

---

## Step 1: Install Arduino Libraries

Open Arduino IDE and install these libraries via **Tools → Manage Libraries**:

### 1.1 Arduino_TensorFlowLite
- Search: "Arduino_TensorFlowLite"
- By: TensorFlow Authors
- Version: Latest stable
- Click **Install**

### 1.2 ArduCAM
- Search: "ArduCAM"
- By: Lee
- Version: Latest stable
- Click **Install**

### 1.3 Wire and SPI (Pre-installed)
- These come with Arduino IDE by default

---

## Step 2: Wire ArduCam to Arduino

```
ArduCam OV7675    →    Arduino Nano 33 Sense
----------------------------------------
CS                →    D7
MOSI              →    D11 (SPI MOSI)
MISO              →    D12 (SPI MISO)
SCK               →    D13 (SPI SCK)
SDA               →    A4  (I2C SDA)
SCL               →    A5  (I2C SCL)
VCC               →    3.3V ⚠️ NOT 5V!
GND               →    GND
```

**⚠️ WARNING**: ArduCam OV7675 operates at 3.3V. Do NOT connect to 5V or it will be damaged.

---

## Step 3: Convert Model to C Header

After training and quantizing your model to `.tflite`, run:

```bash
python convert_to_tflite.py \
  --model training_output/best_model.h5 \
  --dataset_dir preprocessed_dataset \
  --output_dir tflite_models
```

This generates `model_data.h` in the `tflite_models/` directory.

---

## Step 4: Copy Files to Arduino Sketch

1. Create a new folder: `arduino_person_detection/`
2. Copy these files into it:
   - `arduino_person_detection.ino` (main firmware)
   - `model_data.h` (from tflite_models/)

Your folder structure:
```
arduino_person_detection/
├── arduino_person_detection.ino
└── model_data.h
```

---

## Step 5: Configure Arduino IDE

### 5.1 Select Board
1. Connect Arduino via USB
2. Go to **Tools → Board → Arduino Mbed OS Nano Boards → Arduino Nano 33 BLE**
3. Go to **Tools → Port** → Select your Arduino's COM port

### 5.2 Verify Board Connection
- Open **Tools → Get Board Info**
- Should display "Arduino Nano 33 BLE" and serial number

---

## Step 6: Adjust Memory Settings (If Needed)

If you get compilation errors about memory, edit `arduino_person_detection.ino`:

### Option A: Reduce Tensor Arena Size
```cpp
constexpr int kTensorArenaSize = 100 * 1024;  // Reduce from 120 KB to 100 KB
```

### Option B: Switch to Grayscale
```cpp
#define IMAGE_CHANNELS 1  // Change from 3 (RGB) to 1 (Grayscale)
```

### Option C: Reduce Input Resolution
```cpp
#define IMAGE_WIDTH 64   // Change from 96
#define IMAGE_HEIGHT 64  // Change from 96
```

**Note**: If you change input dimensions, you must retrain the model with matching dimensions.

---

## Step 7: Compile and Upload

### 7.1 Compile
1. Click **Verify** button (✓) in Arduino IDE
2. Wait for compilation (may take 2-3 minutes)
3. Check for errors in output console

### 7.2 Common Compilation Errors

**Error: "model_data.h: No such file"**
- Solution: Ensure `model_data.h` is in the same folder as `.ino` file

**Error: "region `FLASH' overflowed"**
- Solution: Model is too large. Reduce model size or use smaller architecture

**Error: "region `RAM' overflowed"**
- Solution: Reduce `kTensorArenaSize` or use grayscale input

### 7.3 Upload
1. Click **Upload** button (→) in Arduino IDE
2. Wait for upload (30-60 seconds)
3. Arduino will reset automatically

---

## Step 8: Monitor Output

### 8.1 Open Serial Monitor
1. Go to **Tools → Serial Monitor**
2. Set baud rate to **115200**
3. You should see initialization messages:

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

[Frame 1] no_person (67.3%) | Inference: 145 ms | FPS: 6.9
[Frame 2] person (82.1%) | Inference: 143 ms | FPS: 7.0
[Frame 3] person (91.5%) | Inference: 144 ms | FPS: 6.9
```

### 8.2 Interpret Output

Each line shows:
- **Frame number**: Sequential frame count
- **Prediction**: "person" or "no_person"
- **Confidence**: Prediction confidence (0-100%)
- **Inference time**: Time to run model (milliseconds)
- **FPS**: Frames per second (1000 / inference_time)

---

## Step 9: Troubleshooting

### Camera Not Detected
**Symptom**: "ERROR: Camera not detected"

**Solutions**:
1. Check wiring (especially CS pin)
2. Verify 3.3V power supply
3. Try adding delay in `setup()`: `delay(2000);` before camera init
4. Check I2C address: Run I2C scanner sketch

### Low FPS
**Symptom**: FPS < 5

**Solutions**:
1. Reduce input resolution (96x96 → 64x64)
2. Switch to grayscale (3 channels → 1 channel)
3. Use smaller model architecture ("tiny" instead of "small")
4. Optimize ArduCam capture settings

### Memory Errors During Runtime
**Symptom**: Arduino crashes or resets

**Solutions**:
1. Reduce tensor arena size
2. Check for stack overflow (avoid large local variables)
3. Monitor memory with `printMemoryUsage()`

### Low Accuracy
**Symptom**: Many incorrect predictions

**Solutions**:
1. Ensure proper lighting conditions
2. Recalibrate camera exposure
3. Retrain model with more diverse data
4. Check preprocessing matches training pipeline

---

## Step 10: Production Optimization

### 10.1 Disable Serial Debugging
Comment out Serial prints in production to save memory and improve speed:

```cpp
// Serial.println("Frame captured");  // Commented out
```

### 10.2 Use MicroMutableOpResolver
Replace `AllOpsResolver` with specific ops to save ~20-30 KB:

```cpp
static tflite::MicroMutableOpResolver<6> micro_op_resolver;
micro_op_resolver.AddConv2D();
micro_op_resolver.AddMaxPool2D();
micro_op_resolver.AddFullyConnected();
micro_op_resolver.AddReshape();
micro_op_resolver.AddSoftmax();
micro_op_resolver.AddQuantize();  // If using INT8
```

### 10.3 Power Optimization
Add sleep modes between inferences:

```cpp
#include <ArduinoBLE.h>

void loop() {
  // Run inference
  runInference();
  
  // Sleep for 100ms
  delay(100);  // Or use deep sleep for battery operation
}
```

---

## Performance Benchmarks

### Expected Performance (96x96 RGB, Small Model)
- **Inference Time**: 100-150 ms
- **FPS**: 6-10
- **Accuracy**: 75-85%
- **RAM Usage**: 140-180 KB
- **Flash Usage**: 130-180 KB

### Expected Performance (64x64 Grayscale, Tiny Model)
- **Inference Time**: 50-80 ms
- **FPS**: 12-20
- **Accuracy**: 70-80%
- **RAM Usage**: 100-130 KB
- **Flash Usage**: 80-120 KB

---

## Additional Resources

- **TensorFlow Lite Micro**: https://www.tensorflow.org/lite/microcontrollers
- **ArduCam Documentation**: http://www.arducam.com/downloads/
- **Arduino Nano 33 Sense**: https://docs.arduino.cc/hardware/nano-33-ble-sense

---

## Next Steps

1. **Optimize lighting**: Test in different lighting conditions
2. **Collect edge cases**: Add failures to training data
3. **Implement action**: Trigger LED, buzzer, or relay on detection
4. **Log data**: Save detections to SD card or send via BLE
5. **Battery operation**: Add LiPo battery and power management

Good luck with your TinyML person detection system! 🚀
