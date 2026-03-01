# A. Hardware Wiring Guide

## ArduCam Mini 2MP OV2640 + Arduino Nano 33 BLE Sense Rev2

### Hardware Specifications
- **Camera**: ArduCam Mini 2MP OV2640 with 8MB FIFO (W25Q64JV)
- **MCU**: Arduino Nano 33 BLE Sense Rev2
- **Logic Level**: 3.3V (both devices are 3.3V compatible ✓)
- **Interfaces**: SPI (image data) + I2C (camera control)

---

## Wiring Diagram

```
ArduCam OV2640          Arduino Nano 33 BLE Sense Rev2
──────────────          ──────────────────────────────
CS (Chip Select)   ───→  D7  (GPIO, directly)
MOSI               ───→  D11 (SPI MOSI / COPI)
MISO               ───→  D12 (SPI MISO / CIPO)
SCK                ───→  D13 (SPI SCK)
SDA                ───→  A4  (I2C SDA)
SCL                ───→  A5  (I2C SCL)
VCC                ───→  3.3V ⚠️ CRITICAL: Use 3.3V, NOT 5V!
GND                ───→  GND
```

### Optional: Piezo Buzzer (for audio feedback)
```
Piezo Buzzer            Arduino Nano 33 BLE Sense Rev2
────────────            ──────────────────────────────
+ (Positive)       ───→  D6 (PWM capable)
- (Negative/GND)   ───→  GND
```

**Recommended Buzzer**: Passive piezo buzzer (e.g., PKM13EPYH4000-A0)
- Cost: ~$1-2
- Can produce different frequencies for each person

---

## Wiring Sanity Checklist

### Before Powering On:
- [ ] **Voltage Check**: VCC connected to 3.3V (NOT 5V!)
- [ ] **GND Connected**: Common ground between camera and Arduino
- [ ] **CS Pin**: Connected to D7 (configurable in code)
- [ ] **SPI Connections**: MOSI→D11, MISO→D12, SCK→D13
- [ ] **I2C Connections**: SDA→A4, SCL→A5
- [ ] **No shorts**: Check for accidental bridges between pins
- [ ] **Secure connections**: Use jumper wires or solder (no loose connections)

### After Powering On:
- [ ] **Power LED**: Arduino LED lights up
- [ ] **No smoke/heat**: Camera stays cool
- [ ] **Serial test**: Upload basic sketch, check Serial Monitor works
- [ ] **I2C scan**: Run I2C scanner to detect camera (address 0x30 or 0x60)
- [ ] **SPI test**: ArduCam test sketch returns correct chip ID

---

## Pin Reference Table

| Function | ArduCam Pin | Arduino Pin | Notes |
|----------|-------------|-------------|-------|
| Chip Select | CS | D7 | Can change to D10 if needed |
| SPI Data Out | MOSI | D11 | Fixed SPI pin |
| SPI Data In | MISO | D12 | Fixed SPI pin |
| SPI Clock | SCK | D13 | Fixed SPI pin |
| I2C Data | SDA | A4 | Fixed I2C pin |
| I2C Clock | SCL | A5 | Fixed I2C pin |
| Power | VCC | 3.3V | ⚠️ 3.3V ONLY |
| Ground | GND | GND | Common ground |
| Buzzer + | - | D6 | PWM for tones |
| Buzzer - | - | GND | Common ground |

---

## Common Wiring Mistakes

### ❌ WRONG: Using 5V
```
VCC → 5V  ← WILL DAMAGE CAMERA!
```

### ✅ CORRECT: Using 3.3V
```
VCC → 3.3V  ← Safe for OV2640
```

### ❌ WRONG: Swapped SDA/SCL
```
SDA → A5 (SCL pin)
SCL → A4 (SDA pin)
```
Camera won't respond on I2C.

### ✅ CORRECT: Proper I2C
```
SDA → A4
SCL → A5
```

---

## I2C Scanner Test Code

Upload this to verify camera is detected:

```cpp
#include <Wire.h>

void setup() {
  Serial.begin(115200);
  while (!Serial);
  Wire.begin();
  Serial.println("I2C Scanner - Looking for ArduCam...");
}

void loop() {
  byte error, address;
  int nDevices = 0;
  
  for (address = 1; address < 127; address++) {
    Wire.beginTransmission(address);
    error = Wire.endTransmission();
    
    if (error == 0) {
      Serial.print("I2C device found at 0x");
      Serial.println(address, HEX);
      nDevices++;
    }
  }
  
  if (nDevices == 0) {
    Serial.println("No I2C devices found - check wiring!");
  } else {
    Serial.println("Scan complete.");
  }
  
  delay(5000);
}
```

**Expected Output**:
```
I2C device found at 0x30
```
or
```
I2C device found at 0x60
```

If no device found → check SDA/SCL wiring and 3.3V power.

---

## ArduCam SPI Test Code

Upload this to verify SPI communication:

```cpp
#include <Wire.h>
#include <SPI.h>
#include <ArduCAM.h>
#include "memorysaver.h"

#define CS_PIN 7

ArduCAM myCAM(OV2640, CS_PIN);

void setup() {
  Serial.begin(115200);
  while (!Serial);
  
  Wire.begin();
  SPI.begin();
  
  pinMode(CS_PIN, OUTPUT);
  digitalWrite(CS_PIN, HIGH);
  
  // Test SPI
  myCAM.write_reg(ARDUCHIP_TEST1, 0x55);
  uint8_t temp = myCAM.read_reg(ARDUCHIP_TEST1);
  
  if (temp != 0x55) {
    Serial.println("SPI ERROR: Check wiring!");
    Serial.print("Expected: 0x55, Got: 0x");
    Serial.println(temp, HEX);
    while (1);
  }
  
  Serial.println("SPI OK!");
  
  // Check camera module type
  uint8_t vid, pid;
  myCAM.wrSensorReg8_8(0xFF, 0x01);
  myCAM.rdSensorReg8_8(OV2640_CHIPID_HIGH, &vid);
  myCAM.rdSensorReg8_8(OV2640_CHIPID_LOW, &pid);
  
  if ((vid != 0x26) || (pid != 0x42)) {
    Serial.println("Camera not OV2640!");
  } else {
    Serial.println("OV2640 detected!");
  }
}

void loop() {}
```

**Expected Output**:
```
SPI OK!
OV2640 detected!
```

---

## Memory Budget (Arduino Nano 33 BLE Sense Rev2)

| Resource | Total | Stage A Model | Stage B Model | Image Buffer | Tensor Arena | Available |
|----------|-------|---------------|---------------|--------------|--------------|-----------|
| **RAM** | 256 KB | ~1 KB | ~1 KB | 9-12 KB | 80-100 KB | ~140 KB |
| **Flash** | 1 MB | 15-25 KB | 20-35 KB | 0 | 0 | ~940 KB |

### Memory Strategy:
- **Stage A (Person Detection)**: ~20 KB quantized
- **Stage B (Face Recognition)**: ~30 KB quantized
- **Combined Flash**: ~50-55 KB (well under 1 MB limit ✓)
- **Image Buffer**: 96×96×1 = 9,216 bytes (grayscale)
- **Tensor Arena**: Shared between models, ~100 KB max

---

## Power Considerations

- Arduino Nano 33 BLE Sense Rev2: 3.3V logic, powered via USB
- ArduCam OV2640: 3.3V operation
- Current draw: ~150-200mA during capture
- USB provides sufficient power for both

**For battery operation** (optional):
- Use LiPo 3.7V with regulator
- Add sleep mode between captures
- Expected battery life: 4-8 hours with 500mAh LiPo

---

## Next Step

After verifying wiring:
1. Run I2C scanner → confirm camera detected
2. Run SPI test → confirm SPI communication
3. Proceed to data collection (Section B)
