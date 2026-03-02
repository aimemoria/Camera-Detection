# How to Run — TinyML Face Detection System

## Setup on a new laptop

### 1. Install requirements
```bash
# Install Arduino CLI
brew install arduino-cli

# Install Arduino board + libraries
arduino-cli core install arduino:mbed_nano
arduino-cli lib install "ArduCAM"
arduino-cli lib install "Chirale_TensorFLowLite@1.0.1"

# Install Python dependency
pip3 install pyserial
```

### 2. Clone the project
```bash
git clone https://github.com/aimemoria/Camera-Detection.git
cd Camera-Detection
```

### 3. Find your Arduino port
```bash
ls /dev/cu.usbmodem*
# Use whatever port it shows (e.g. /dev/cu.usbmodem1101)
```

### 4. Upload firmware to Arduino
```bash
cd face_recognition_system/G_arduino_firmware
arduino-cli upload --fqbn arduino:mbed_nano:nano33ble --port /dev/cu.usbmodem1101 G_arduino_firmware.ino
```

### 5. Run the live preview
```bash
cd ../   # back to face_recognition_system/
python3 preview_server.py
```

### 6. Open browser
```
http://localhost:7654
```

> **Note:** If your port is different from `usbmodem1101`, edit line 25 of
> `face_recognition_system/preview_server.py` to match.
