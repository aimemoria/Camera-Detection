# H. Testing Protocol and Trial Documentation

## Overview

This document defines the testing protocol for evaluating the face recognition system.
Following the rubric requirements:
- **10 trial tests** (people look at camera)
- **Goal: 8/10 correct** (80% accuracy minimum)
- **Target: 99%+ accuracy** (as specified in requirements)

---

## Test Environment Setup

### Hardware Checklist
- [ ] Arduino Nano 33 BLE Sense Rev2 connected via USB
- [ ] ArduCam Mini 2MP OV2640 properly wired
- [ ] Piezo buzzer connected to D6 (optional for audio)
- [ ] Serial monitor open at 115200 baud
- [ ] Firmware flashed (G_arduino_firmware.ino)

### Environment Conditions
- **Lighting**: Indoor, consistent lighting (note type)
- **Distance**: 0.5-1.5 meters from camera
- **Background**: Note background complexity
- **Time of day**: Record for consistency

---

## Test Categories

### Category 1: Known Person Recognition
Test each of the 5 known persons for correct identification.

### Category 2: Unknown Person Rejection
Test with people NOT in the training set - should output "Unknown".

### Category 3: No Person Detection
Test with empty scenes - should output "No Person".

### Category 4: Edge Cases
Test challenging conditions (poor lighting, partial occlusion, etc.)

---

## Trial Test Template

### Trial Information
| Field | Value |
|-------|-------|
| **Trial Number** | |
| **Date** | |
| **Time** | |
| **Tester Name** | |
| **Environment** | |

---

## Category 1: Known Person Recognition Trials

### Trial 1
| Field | Value |
|-------|-------|
| **Subject** | Person 1 (Name: ___________) |
| **Expected Result** | Person 1 |
| **Actual Result** | |
| **Confidence** | % |
| **Inference Time** | ms |
| **Audio Feedback** | ✓ Correct / ✗ Incorrect |
| **Pass/Fail** | |

**Notes**: 
```
[Observations, lighting conditions, distance, etc.]
```

---

### Trial 2
| Field | Value |
|-------|-------|
| **Subject** | Person 2 (Name: ___________) |
| **Expected Result** | Person 2 |
| **Actual Result** | |
| **Confidence** | % |
| **Inference Time** | ms |
| **Audio Feedback** | ✓ Correct / ✗ Incorrect |
| **Pass/Fail** | |

**Notes**: 
```

```

---

### Trial 3
| Field | Value |
|-------|-------|
| **Subject** | Person 3 (Name: ___________) |
| **Expected Result** | Person 3 |
| **Actual Result** | |
| **Confidence** | % |
| **Inference Time** | ms |
| **Audio Feedback** | ✓ Correct / ✗ Incorrect |
| **Pass/Fail** | |

**Notes**: 
```

```

---

### Trial 4
| Field | Value |
|-------|-------|
| **Subject** | Person 4 (Name: ___________) |
| **Expected Result** | Person 4 |
| **Actual Result** | |
| **Confidence** | % |
| **Inference Time** | ms |
| **Audio Feedback** | ✓ Correct / ✗ Incorrect |
| **Pass/Fail** | |

**Notes**: 
```

```

---

### Trial 5
| Field | Value |
|-------|-------|
| **Subject** | Person 5 (Name: ___________) |
| **Expected Result** | Person 5 |
| **Actual Result** | |
| **Confidence** | % |
| **Inference Time** | ms |
| **Audio Feedback** | ✓ Correct / ✗ Incorrect |
| **Pass/Fail** | |

**Notes**: 
```

```

---

## Category 2: Unknown Person Rejection Trials

### Trial 6
| Field | Value |
|-------|-------|
| **Subject** | Unknown Person A (Name: ___________) |
| **Expected Result** | Unknown |
| **Actual Result** | |
| **Confidence** | % (should be < 70%) |
| **Inference Time** | ms |
| **Audio Feedback** | ✓ Warble / ✗ Wrong pattern |
| **Pass/Fail** | |

**Notes**: 
```
[Person NOT in training set]
```

---

### Trial 7
| Field | Value |
|-------|-------|
| **Subject** | Unknown Person B (Name: ___________) |
| **Expected Result** | Unknown |
| **Actual Result** | |
| **Confidence** | % (should be < 70%) |
| **Inference Time** | ms |
| **Audio Feedback** | ✓ Warble / ✗ Wrong pattern |
| **Pass/Fail** | |

**Notes**: 
```
[Person NOT in training set]
```

---

## Category 3: No Person Detection Trials

### Trial 8
| Field | Value |
|-------|-------|
| **Subject** | Empty Scene (describe: ___________) |
| **Expected Result** | No Person |
| **Actual Result** | |
| **Stage A Output** | [no_person, person] |
| **Inference Time** | ms |
| **Audio Feedback** | ✓ Low beep / ✗ Wrong pattern |
| **Pass/Fail** | |

**Notes**: 
```
[Empty room, wall, desk, etc.]
```

---

## Category 4: Edge Case Trials

### Trial 9
| Field | Value |
|-------|-------|
| **Subject** | Known Person (specify: ___________) |
| **Condition** | (e.g., dim lighting, side angle, partial face) |
| **Expected Result** | Person X |
| **Actual Result** | |
| **Confidence** | % |
| **Inference Time** | ms |
| **Pass/Fail** | |

**Notes**: 
```
[Describe edge case condition]
```

---

### Trial 10
| Field | Value |
|-------|-------|
| **Subject** | (specify: ___________) |
| **Condition** | (e.g., glasses on/off, different expression) |
| **Expected Result** | |
| **Actual Result** | |
| **Confidence** | % |
| **Inference Time** | ms |
| **Pass/Fail** | |

**Notes**: 
```
[Describe edge case condition]
```

---

## Results Summary

### Trial Results Table

| Trial | Subject | Expected | Actual | Confidence | Pass/Fail |
|-------|---------|----------|--------|------------|-----------|
| 1 | Person 1 | Person 1 | | | |
| 2 | Person 2 | Person 2 | | | |
| 3 | Person 3 | Person 3 | | | |
| 4 | Person 4 | Person 4 | | | |
| 5 | Person 5 | Person 5 | | | |
| 6 | Unknown A | Unknown | | | |
| 7 | Unknown B | Unknown | | | |
| 8 | Empty | No Person | | | |
| 9 | Edge Case | | | | |
| 10 | Edge Case | | | | |

---

### Performance Metrics

| Metric | Value | Target |
|--------|-------|--------|
| **Total Trials** | 10 | 10 |
| **Passed** | | ≥8 |
| **Failed** | | ≤2 |
| **Accuracy** | % | ≥80% (target: 99%) |

### Category Breakdown

| Category | Trials | Passed | Accuracy |
|----------|--------|--------|----------|
| Known Person Recognition | 5 | | % |
| Unknown Person Rejection | 2 | | % |
| No Person Detection | 1 | | % |
| Edge Cases | 2 | | % |

### Average Metrics

| Metric | Value |
|--------|-------|
| **Avg Confidence (Known Persons)** | % |
| **Avg Confidence (Unknown)** | % |
| **Avg Inference Time** | ms |

---

## Expanded Testing (Optional)

For more rigorous testing, expand to 50+ trials:

### Known Person Extended Tests (30 trials)
- 6 trials per known person
- Varying: angles, lighting, distance, expressions

### Unknown Person Extended Tests (10 trials)
- 5 different unknown people
- 2 trials each

### Edge Case Extended Tests (10 trials)
- Poor lighting
- Side angles (30°, 45°)
- Partial occlusion
- With/without glasses
- Different expressions

---

## Troubleshooting Common Issues

### Issue: Low Confidence on Known Persons

**Symptoms**: Confidence < 70% for known person
**Possible Causes**:
1. Poor lighting during test vs training
2. Significant angle difference
3. Insufficient training data variety

**Solutions**:
1. Test in similar lighting to training
2. Face camera directly
3. Retrain with more diverse data

---

### Issue: Unknown Person Classified as Known

**Symptoms**: Unknown person gets high confidence for known person
**Possible Causes**:
1. Unknown person looks similar to known person
2. Threshold too low
3. Insufficient training diversity

**Solutions**:
1. Increase unknown threshold (e.g., 0.75 → 0.80)
2. Add more unknown test data for threshold tuning
3. Retrain with more background variety

---

### Issue: Known Person Classified as Unknown

**Symptoms**: Known person gets confidence < threshold
**Possible Causes**:
1. Test conditions very different from training
2. Threshold too high
3. Insufficient training data for that person

**Solutions**:
1. Test in conditions similar to training
2. Lower unknown threshold slightly
3. Add more training images for that person

---

### Issue: Stage A Fails (No Person Detected)

**Symptoms**: Stage A returns "No Person" when person present
**Possible Causes**:
1. Poor image quality
2. Person too far from camera
3. Insufficient person training data

**Solutions**:
1. Check camera focus and lighting
2. Move closer to camera (0.5-1m)
3. Retrain Stage A with more person data

---

## Serial Output Reference

### Successful Known Person Detection
```
--- Starting Inference ---
1. Capturing image...
   Captured 12456 bytes
2. Preprocessing...
   Preprocessing complete
3. Running Stage A (Person Detection)...
   Stage A output: [0.023, 0.977] (45 ms)
   Person detected! Running Stage B...
4. Running Stage B (Face Recognition)...
   Stage B output: [0.891, 0.042, 0.031, 0.019, 0.017] (78 ms)

========================================
RESULT: Person 1 (ID: 1)
Confidence: 89.1%
Inference time: 156 ms
========================================
```

### Successful Unknown Detection
```
--- Starting Inference ---
1. Capturing image...
   Captured 11234 bytes
2. Preprocessing...
   Preprocessing complete
3. Running Stage A (Person Detection)...
   Stage A output: [0.031, 0.969] (45 ms)
   Person detected! Running Stage B...
4. Running Stage B (Face Recognition)...
   Stage B output: [0.234, 0.198, 0.187, 0.201, 0.180] (78 ms)

========================================
RESULT: UNKNOWN PERSON
Confidence: 23.4% (below threshold)
Inference time: 158 ms
========================================
```

### No Person Detected
```
--- Starting Inference ---
1. Capturing image...
   Captured 8765 bytes
2. Preprocessing...
   Preprocessing complete
3. Running Stage A (Person Detection)...
   Stage A output: [0.934, 0.066] (45 ms)

Result: NO PERSON DETECTED
```

---

## Sign-Off

| Role | Name | Signature | Date |
|------|------|-----------|------|
| **Tester** | | | |
| **Reviewer** | | | |
| **Approver** | | | |

---

## Appendix: Audio Feedback Reference

| Result | Beep Pattern | Description |
|--------|--------------|-------------|
| No Person | 1 low beep | 200Hz, 100ms |
| Person 1 | 2 high beeps | 1000Hz × 2, 100ms each |
| Person 2 | 1 long mid beep | 800Hz, 300ms |
| Person 3 | 3 ascending beeps | 600→800→1000Hz |
| Person 4 | 2 descending beeps | 1000→600Hz |
| Person 5 | Short + long | 900Hz: 80ms + 200ms |
| Unknown | Low warble | 300→400→300Hz |
