# B. Data Collection Protocol

## Overview

This protocol ensures high-quality training data for:
- **Stage A**: Person vs No-Person (Binary)
- **Stage B**: 5 Known Persons (Multi-class)
- **Unknown Detection**: Via confidence thresholding (NOT a separate class)

---

## Dataset Structure

```
dataset/
├── stage_a/                          # Person Detection data
│   ├── person/                       # Images with ANY person
│   │   ├── person1_001.jpg
│   │   ├── person1_002.jpg
│   │   ├── person2_001.jpg
│   │   └── ... (all 5 persons combined)
│   └── no_person/                    # Background, empty scenes
│       ├── background_001.jpg
│       ├── background_002.jpg
│       └── ...
│
├── stage_b/                          # Face Recognition data
│   ├── person1/                      # Individual: Person 1
│   │   ├── p1_front_001.jpg
│   │   ├── p1_left_001.jpg
│   │   └── ...
│   ├── person2/                      # Individual: Person 2
│   │   └── ...
│   ├── person3/                      # Individual: Person 3
│   │   └── ...
│   ├── person4/                      # Individual: Person 4
│   │   └── ...
│   └── person5/                      # Individual: Person 5
│       └── ...
│
└── unknown_test/                     # For threshold tuning (NOT training)
    ├── other_person1/                # People NOT in person1-5
    │   └── ...
    ├── other_person2/
    │   └── ...
    └── other_person3/
        └── ...
```

---

## Minimum Image Requirements

### Stage A (Person Detection)
| Class | Minimum | Recommended | Notes |
|-------|---------|-------------|-------|
| **person** | 500 | 1,000 | Combined from all 5 known persons |
| **no_person** | 500 | 1,000 | Empty backgrounds, objects |

### Stage B (Face Recognition)
| Class | Minimum | Recommended | Notes |
|-------|---------|-------------|-------|
| **person1** | 100 | 200 | Each known individual |
| **person2** | 100 | 200 | |
| **person3** | 100 | 200 | |
| **person4** | 100 | 200 | |
| **person5** | 100 | 200 | |

### Unknown Test Set (for threshold tuning)
| Class | Minimum | Notes |
|-------|---------|-------|
| **other_person1** | 50 | Person NOT in training set |
| **other_person2** | 50 | Different person NOT in training |
| **other_person3** | 50 | Third person NOT in training |

**CRITICAL**: Unknown persons are ONLY used for threshold tuning, NOT for training. The model learns 5 classes; "Unknown" is detected via low confidence.

---

## Image Capture Protocol

### Equipment Needed
- ArduCam OV2640 connected to Arduino
- OR smartphone camera (then resize to 96×96)
- Consistent capture distance: 0.5-1.5 meters
- Multiple locations with varying backgrounds

### Capture Settings
```
Resolution: Capture at higher res, resize to 96×96
Format: JPEG or PNG
Color: RGB (will convert to grayscale in preprocessing)
Naming: {class}_{variation}_{number}.jpg
Example: person1_front_bright_001.jpg
```

---

## Per-Person Capture Checklist

### For Each of the 5 Known Persons:

#### Orientations (capture 15-20 images each)
- [ ] **Front** (looking directly at camera)
- [ ] **Left 15°** (slight left turn)
- [ ] **Left 30°** (moderate left turn)
- [ ] **Left 45°** (significant left turn)
- [ ] **Right 15°**
- [ ] **Right 30°**
- [ ] **Right 45°**
- [ ] **Up 10°** (chin slightly up)
- [ ] **Down 10°** (chin slightly down)

#### Distances (capture 10-15 images each)
- [ ] **Close**: 0.5 meters (face fills ~70% of frame)
- [ ] **Medium**: 1.0 meters (face fills ~40% of frame)
- [ ] **Far**: 1.5 meters (face fills ~25% of frame)

#### Lighting Conditions (capture 10-15 images each)
- [ ] **Bright indoor** (overhead lights on)
- [ ] **Dim indoor** (lights off, window light only)
- [ ] **Natural daylight** (near window)
- [ ] **Fluorescent light** (office environment)
- [ ] **Mixed lighting** (multiple sources)

#### Backgrounds (capture 10-15 images each)
- [ ] **Plain wall** (white, beige, or solid color)
- [ ] **Classroom** (desks, chairs visible)
- [ ] **Hallway** (corridor environment)
- [ ] **Outdoor** (if applicable)
- [ ] **Cluttered** (complex background)

#### Expressions (capture 5-10 images each)
- [ ] **Neutral**
- [ ] **Smiling**
- [ ] **Slight frown**

#### Accessories Variations (capture 5-10 images each)
- [ ] **No glasses** (if applicable)
- [ ] **With glasses** (if applicable)
- [ ] **With hat** (optional)
- [ ] **Different hairstyle** (if varies day to day)

### Minimum Total Per Person: 100 images
### Recommended Total Per Person: 200 images

---

## No-Person (Background) Capture Protocol

### What to Capture:
- [ ] **Empty classroom** (same room as testing)
- [ ] **Empty hallway**
- [ ] **Walls** (plain, with posters, with windows)
- [ ] **Desks/chairs** (without people)
- [ ] **Outdoor scenes** (if testing outdoors)
- [ ] **Various objects** (but no humans visible)

### What to AVOID:
- ❌ Images with any person visible (even partially)
- ❌ Images with photos/posters of people
- ❌ Mannequins or human-like objects

### Minimum: 500 images
### Recommended: 1,000 images

---

## Unknown Person Capture Protocol

### Purpose:
Tune the confidence threshold so unknown persons are NOT misclassified as known persons.

### Who to Capture:
- 3+ people who are **NOT** in the known person1-5 set
- Classmates who will NOT be tested as known persons
- Friends, family members (if available)

### How Many:
- 50 images per unknown person minimum
- Total: 150+ unknown person images

### How to Use:
1. **NOT** for training the model
2. Used ONLY to tune the confidence threshold
3. Feed through trained model
4. Adjust threshold until unknown persons get "Unknown" prediction

---

## Image Naming Convention

```
{class}_{angle}_{lighting}_{distance}_{number}.jpg

Examples:
person1_front_bright_close_001.jpg
person1_left30_dim_medium_012.jpg
person2_right15_natural_far_007.jpg
no_person_classroom_bright_023.jpg
unknown1_front_bright_close_005.jpg
```

---

## Capture Script (Arduino)

Use this sketch to capture images directly from ArduCam:

```cpp
// capture_training_images.ino
// Captures JPEG images and sends via Serial for saving on PC

#include <Wire.h>
#include <SPI.h>
#include <ArduCAM.h>
#include "memorysaver.h"

#define CS_PIN 7

ArduCAM myCAM(OV2640, CS_PIN);

void setup() {
  Serial.begin(921600);  // High baud for image transfer
  while (!Serial);
  
  Wire.begin();
  SPI.begin();
  
  pinMode(CS_PIN, OUTPUT);
  digitalWrite(CS_PIN, HIGH);
  
  // Initialize camera
  myCAM.write_reg(ARDUCHIP_TEST1, 0x55);
  if (myCAM.read_reg(ARDUCHIP_TEST1) != 0x55) {
    Serial.println("SPI ERROR");
    while (1);
  }
  
  myCAM.set_format(JPEG);
  myCAM.InitCAM();
  myCAM.OV2640_set_JPEG_size(OV2640_160x120);  // Small for faster transfer
  myCAM.clear_fifo_flag();
  
  Serial.println("READY");
  Serial.println("Send 'c' to capture image");
}

void loop() {
  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == 'c' || cmd == 'C') {
      captureAndSend();
    }
  }
}

void captureAndSend() {
  myCAM.flush_fifo();
  myCAM.clear_fifo_flag();
  myCAM.start_capture();
  
  while (!myCAM.get_bit(ARDUCHIP_TRIG, CAP_DONE_MASK));
  
  uint32_t length = myCAM.read_fifo_length();
  
  if (length >= 0x5FFFF) {
    Serial.println("ERROR: Image too large");
    return;
  }
  
  if (length == 0) {
    Serial.println("ERROR: FIFO empty");
    return;
  }
  
  Serial.print("IMG_START:");
  Serial.println(length);
  
  myCAM.CS_LOW();
  myCAM.set_fifo_burst();
  
  for (uint32_t i = 0; i < length; i++) {
    uint8_t data = SPI.transfer(0x00);
    Serial.write(data);
  }
  
  myCAM.CS_HIGH();
  Serial.println();
  Serial.println("IMG_END");
}
```

---

## Python Script to Save Captured Images

```python
#!/usr/bin/env python3
"""
capture_from_arduino.py
Receives images from Arduino and saves them to dataset folder.

Usage:
  python capture_from_arduino.py --class person1 --port /dev/cu.usbmodem14101
"""

import serial
import os
import argparse
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Capture images from Arduino')
    parser.add_argument('--class_name', type=str, required=True,
                        help='Class name: person1, person2, ..., no_person')
    parser.add_argument('--port', type=str, required=True,
                        help='Serial port (e.g., /dev/cu.usbmodem14101 or COM3)')
    parser.add_argument('--output_dir', type=str, default='dataset',
                        help='Output directory')
    parser.add_argument('--count', type=int, default=100,
                        help='Number of images to capture')
    args = parser.parse_args()
    
    # Create output directory
    if args.class_name in ['person1', 'person2', 'person3', 'person4', 'person5']:
        output_path = os.path.join(args.output_dir, 'stage_b', args.class_name)
    elif args.class_name == 'no_person':
        output_path = os.path.join(args.output_dir, 'stage_a', 'no_person')
    elif args.class_name.startswith('unknown'):
        output_path = os.path.join(args.output_dir, 'unknown_test', args.class_name)
    else:
        output_path = os.path.join(args.output_dir, args.class_name)
    
    os.makedirs(output_path, exist_ok=True)
    
    # Connect to Arduino
    print(f"Connecting to {args.port}...")
    ser = serial.Serial(args.port, 921600, timeout=10)
    
    # Wait for ready
    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        print(line)
        if 'READY' in line:
            break
    
    print(f"\nCapturing {args.count} images for class '{args.class_name}'")
    print(f"Saving to: {output_path}")
    print("Press Enter to capture each image, 'q' to quit\n")
    
    captured = 0
    while captured < args.count:
        user_input = input(f"[{captured+1}/{args.count}] Press Enter to capture: ")
        if user_input.lower() == 'q':
            break
        
        # Send capture command
        ser.write(b'c')
        
        # Read response
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line.startswith('IMG_START:'):
                img_size = int(line.split(':')[1])
                break
            elif 'ERROR' in line:
                print(f"Error: {line}")
                break
        
        # Read image data
        img_data = ser.read(img_size)
        
        # Wait for IMG_END
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if 'IMG_END' in line:
                break
        
        # Save image
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{args.class_name}_{timestamp}_{captured+1:03d}.jpg"
        filepath = os.path.join(output_path, filename)
        
        with open(filepath, 'wb') as f:
            f.write(img_data)
        
        print(f"  Saved: {filename} ({img_size} bytes)")
        captured += 1
    
    print(f"\nCapture complete! {captured} images saved to {output_path}")
    ser.close()

if __name__ == '__main__':
    main()
```

---

## Alternative: Smartphone Capture

If Arduino capture is too slow, use smartphone:

### Steps:
1. Take photos with smartphone camera
2. Ensure face is centered and well-lit
3. Transfer to computer
4. Run preprocessing script (Section C) to resize to 96×96

### Smartphone Tips:
- Hold phone at arm's length (~0.5-1m from face)
- Use burst mode for quick captures
- Capture in landscape OR portrait (preprocessing will crop)
- Name files according to convention before moving to dataset folder

---

## Data Collection Schedule

### Day 1: Known Persons (2-3 hours)
- [ ] Capture Person1: 100-200 images (30 min)
- [ ] Capture Person2: 100-200 images (30 min)
- [ ] Capture Person3: 100-200 images (30 min)
- [ ] Capture Person4: 100-200 images (30 min)
- [ ] Capture Person5: 100-200 images (30 min)

### Day 2: Backgrounds + Unknown (1-2 hours)
- [ ] Capture no_person backgrounds: 500-1000 images (45 min)
- [ ] Capture unknown_person1: 50 images (15 min)
- [ ] Capture unknown_person2: 50 images (15 min)
- [ ] Capture unknown_person3: 50 images (15 min)

### Day 3: Data Verification
- [ ] Review all images for quality
- [ ] Remove blurry/incorrect images
- [ ] Ensure balanced class distribution
- [ ] Run preprocessing pipeline

---

## Quality Checklist

Before proceeding to training:

- [ ] **Face visible**: Face clearly visible in each image
- [ ] **Centered**: Face roughly centered in frame
- [ ] **Not blurry**: Images are sharp and in focus
- [ ] **Correct class**: Each image in correct folder
- [ ] **Minimum count**: At least 100 images per known person
- [ ] **Diversity**: Multiple angles/lighting per person
- [ ] **No duplicates**: No exact duplicate images
- [ ] **Background variety**: Multiple backgrounds per person

---

## Summary Table

| Dataset | Class | Purpose | Minimum | Used For |
|---------|-------|---------|---------|----------|
| stage_a | person | Binary detection | 500 | Training Stage A |
| stage_a | no_person | Binary detection | 500 | Training Stage A |
| stage_b | person1 | Identity | 100 | Training Stage B |
| stage_b | person2 | Identity | 100 | Training Stage B |
| stage_b | person3 | Identity | 100 | Training Stage B |
| stage_b | person4 | Identity | 100 | Training Stage B |
| stage_b | person5 | Identity | 100 | Training Stage B |
| unknown_test | other_person* | Threshold tuning | 150 | Validation ONLY |

**Total Minimum Images**: ~1,650
**Recommended**: ~3,000+
