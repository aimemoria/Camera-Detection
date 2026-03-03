/*
 * G. Arduino Firmware — Face Detection Only
 *
 * Target Hardware:
 *   - Arduino Nano 33 BLE Sense Rev2
 *   - ArduCam Mini 2MP OV2640 with 8MB FIFO
 *   - Piezo buzzer on pin D6  (optional, for audio feedback)
 *
 * Pipeline:
 *   Stage A: Person Detection — binary: person vs no_person
 *
 * Flow:
 *   1. Capture 96x96 grayscale image from ArduCam
 *   2. Run Stage A → no person? → Serial + 1 short beep, stop
 *                  → person?    → Serial + 2 beeps
 *
 * Serial feedback format (115200 baud):
 *   RESULT: Face Detected
 *   CONFIDENCE: 95%
 *
 *   RESULT: No face detected
 *   CONFIDENCE: ---
 *
 * Buzzer patterns:
 *   No person — 1 short low beep  (200 Hz)
 *   Person    — 2 short high beeps (1000 Hz)
 */

// =============================================================================
// Libraries
// =============================================================================
#include <Wire.h>
#include <SPI.h>
#include <ArduCAM.h>
#include "memorysaver.h"

// mbed defines swap(a,b,size) as a 3-arg macro; undefine it so
// the TFLite/STL std::swap templates compile without conflict.
#ifdef swap
#undef swap
#endif

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "stage_a_model.h"

// =============================================================================
// Configuration
// =============================================================================
#define CS_PIN              7       // ArduCam chip-select
#define BUZZER_PIN          6       // Piezo buzzer (optional)

#define IMG_WIDTH           96
#define IMG_HEIGHT          96
#define IMG_CHANNELS        1

#define ARENA_A_SIZE  (48 * 1024)   // 48 KB for Stage A
#define INFERENCE_INTERVAL_MS  500  // ms between frames

// =============================================================================
// Buzzer patterns  {freq, duration_ms, pause_ms, ...}  — zero-terminated
// =============================================================================
const int BEEP_PATTERNS[][10] = {
  // 0: No person — single low beep
  {200,  100, 0,    0,    0,  0, 0, 0, 0, 0},
  // 1: Person    — two short high beeps
  {1000, 100, 50, 1000, 100,  0, 0, 0, 0, 0},
};

#define PATTERN_NO_PERSON   0
#define PATTERN_PERSON      1

// =============================================================================
// Global Variables
// =============================================================================
ArduCAM myCAM(OV2640, CS_PIN);

bool camera_ok = false;

const tflite::Model*      model_stage_a = nullptr;
tflite::MicroInterpreter* interp_a      = nullptr;

alignas(16) uint8_t tensor_arena_a[ARENA_A_SIZE];
uint8_t image_buffer[IMG_WIDTH * IMG_HEIGHT];

unsigned long last_inference_time = 0;

// =============================================================================
// Forward declarations
// =============================================================================
void setupCamera();
void setupTFLite();
bool captureImage();
void restoreJPEGMode();
void preprocessImage();
int  runStageA();
void playBeepPattern(int pattern_index);
void printSeparator();
void streamPreviewJPEG();

// =============================================================================
// Setup
// =============================================================================
void setup() {
  Serial.begin(115200);
  while (!Serial);

  Serial.println("\n========================================");
  Serial.println("  TinyML Face Detection System");
  Serial.println("  Arduino Nano 33 BLE Sense Rev2");
  Serial.println("========================================");

  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);
  tone(BUZZER_PIN, 1000, 100); delay(150);
  tone(BUZZER_PIN, 1200, 100); delay(150);
  tone(BUZZER_PIN, 1500, 150); delay(200);

  Wire.begin();
  SPI.begin();

  Serial.println("Initialising camera ...");
  setupCamera();

  Serial.println("Initialising TensorFlow Lite ...");
  setupTFLite();

  Serial.println("\nSystem ready — detecting ...\n");
}

// =============================================================================
// Main Loop
// =============================================================================
void loop() {
  unsigned long now = millis();
  if (now - last_inference_time >= INFERENCE_INTERVAL_MS) {
    last_inference_time = now;
    runInference();
    if (camera_ok) streamPreviewJPEG();
  }
}

// =============================================================================
// Inference Pipeline — Stage A only
// =============================================================================
void runInference() {
  printSeparator();
  unsigned long t0 = millis();

  if (!captureImage()) {
    Serial.println("ERROR: Camera capture failed");
    playBeepPattern(PATTERN_NO_PERSON);
    return;
  }
  preprocessImage();

  int stage_a = runStageA();

  if (stage_a <= 0) {
    Serial.println("RESULT: No face detected");
    Serial.println("CONFIDENCE: ---");
    playBeepPattern(PATTERN_NO_PERSON);
  } else {
    // Read confidence from Stage A output for reporting
    TfLiteTensor* output = interp_a->output(0);
    float p_yes;
    if (output->type == kTfLiteInt8) {
      float s   = output->params.scale;
      int32_t z = output->params.zero_point;
      p_yes = (output->data.int8[1] - z) * s;
    } else {
      p_yes = output->data.f[1];
    }
    Serial.println("RESULT: Face Detected");
    Serial.print("CONFIDENCE: ");
    Serial.print((int)(p_yes * 100.0f));
    Serial.println("%");
    playBeepPattern(PATTERN_PERSON);
  }

  Serial.print("Inference time: ");
  Serial.print(millis() - t0);
  Serial.println(" ms");
  printSeparator();
}

// =============================================================================
// Camera Setup
// =============================================================================
void setupCamera() {
  pinMode(CS_PIN, OUTPUT);
  digitalWrite(CS_PIN, HIGH);

  myCAM.write_reg(ARDUCHIP_TEST1, 0x55);
  if (myCAM.read_reg(ARDUCHIP_TEST1) != 0x55) {
    Serial.println("WARNING: SPI interface test failed - camera unavailable");
    Serial.println(">>> Running in TEST MODE with synthetic image <<<");
    camera_ok = false;
    return;
  }

  uint8_t vid, pid;
  myCAM.wrSensorReg8_8(0xFF, 0x01);
  myCAM.rdSensorReg8_8(OV2640_CHIPID_HIGH, &vid);
  myCAM.rdSensorReg8_8(OV2640_CHIPID_LOW,  &pid);

  if (vid != 0x26 || (pid != 0x41 && pid != 0x42)) {
    Serial.println("WARNING: OV2640 not detected - camera unavailable");
    Serial.println(">>> Running in TEST MODE with synthetic image <<<");
    camera_ok = false;
    return;
  }

  Serial.println("Camera: OV2640 OK");
  myCAM.set_format(JPEG);
  myCAM.InitCAM();
  myCAM.OV2640_set_JPEG_size(OV2640_320x240);
  delay(100);
  myCAM.clear_fifo_flag();
  camera_ok = true;
}

// =============================================================================
// TFLite Setup — Stage A only
// =============================================================================
void setupTFLite() {
  static tflite::AllOpsResolver resolver;

  model_stage_a = tflite::GetModel(stage_a_model);
  if (model_stage_a->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERROR: Stage A schema mismatch!"); while (1);
  }
  Serial.print("Stage A model: ");
  Serial.print(stage_a_model_len);
  Serial.println(" bytes in flash");

  static tflite::MicroInterpreter si_a(
    model_stage_a, resolver, tensor_arena_a, ARENA_A_SIZE);
  interp_a = &si_a;
  if (interp_a->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERROR: Stage A tensor allocation failed!"); while (1);
  }
  Serial.print("Stage A arena used: ");
  Serial.print(interp_a->arena_used_bytes());
  Serial.print(" / ");
  Serial.print(ARENA_A_SIZE);
  Serial.println(" bytes");
}

// =============================================================================
// Image Capture
// =============================================================================
void restoreJPEGMode() {
  myCAM.set_format(JPEG);
  myCAM.InitCAM();
  myCAM.OV2640_set_JPEG_size(OV2640_320x240);
  delay(100);
  myCAM.clear_fifo_flag();
}

bool captureImage() {
  if (!camera_ok) {
    memset(image_buffer, 128, IMG_WIDTH * IMG_HEIGHT);
    return true;
  }

  myCAM.set_format(BMP);
  myCAM.InitCAM();
  myCAM.OV2640_set_JPEG_size(OV2640_160x120);
  delay(100);

  myCAM.flush_fifo();
  myCAM.clear_fifo_flag();
  myCAM.start_capture();

  unsigned long t = millis();
  while (!myCAM.get_bit(ARDUCHIP_TRIG, CAP_DONE_MASK)) {
    if (millis() - t > 3000) {
      myCAM.CS_HIGH();
      restoreJPEGMode();
      Serial.println("Capture timeout");
      return false;
    }
  }

  const int SRC_W    = 160, SRC_H = 120;
  const int COL_SKIP = (SRC_W - IMG_WIDTH)  / 2;
  const int ROW_SKIP = (SRC_H - IMG_HEIGHT) / 2;

  myCAM.CS_LOW();
  myCAM.set_fifo_burst();

  int out = 0;
  for (int r = 0; r < SRC_H; r++) {
    bool in_row = (r >= ROW_SKIP && r < ROW_SKIP + IMG_HEIGHT);
    for (int c = 0; c < SRC_W; c++) {
      uint8_t hi = SPI.transfer(0x00);
      uint8_t lo = SPI.transfer(0x00);
      if (in_row && c >= COL_SKIP && c < COL_SKIP + IMG_WIDTH) {
        uint8_t r8 = (hi & 0xF8);
        uint8_t g8 = ((hi & 0x07) << 5) | ((lo & 0xE0) >> 3);
        uint8_t b8 = (lo & 0x1F) << 3;
        image_buffer[out++] = (uint8_t)((77u * r8 + 150u * g8 + 29u * b8) >> 8);
      }
    }
  }

  myCAM.CS_HIGH();
  restoreJPEGMode();
  return (out == IMG_WIDTH * IMG_HEIGHT);
}

// =============================================================================
// Preprocessing
// =============================================================================
void preprocessImage() {
  for (int i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++) {
    if (image_buffer[i] > 255) image_buffer[i] = 255;
  }
}

// =============================================================================
// Fill TFLite input tensor from image_buffer
// =============================================================================
static void fillInputTensor(TfLiteTensor* input) {
  if (input->type == kTfLiteInt8) {
    float   scale   = input->params.scale;
    int32_t zero_pt = input->params.zero_point;
    int8_t* data    = input->data.int8;
    for (int i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++) {
      float   norm = image_buffer[i] / 255.0f;
      int32_t q    = (int32_t)(norm / scale) + zero_pt;
      if (q >  127) q =  127;
      if (q < -128) q = -128;
      data[i] = (int8_t)q;
    }
  } else {
    float* data = input->data.f;
    for (int i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++) {
      data[i] = image_buffer[i] / 255.0f;
    }
  }
}

// =============================================================================
// Stage A — Person detection
// Returns 1 (person), 0 (no person), -1 (error)
// =============================================================================
int runStageA() {
  TfLiteTensor* input  = interp_a->input(0);
  TfLiteTensor* output = interp_a->output(0);

  fillInputTensor(input);

  unsigned long t = millis();
  if (interp_a->Invoke() != kTfLiteOk) {
    Serial.println("Stage A invoke failed"); return -1;
  }
  Serial.print("Stage A: ");
  Serial.print(millis() - t);
  Serial.print(" ms — ");

  float p_no, p_yes;
  if (output->type == kTfLiteInt8) {
    float s   = output->params.scale;
    int32_t z = output->params.zero_point;
    p_no  = (output->data.int8[0] - z) * s;
    p_yes = (output->data.int8[1] - z) * s;
  } else {
    p_no  = output->data.f[0];
    p_yes = output->data.f[1];
  }

  if (p_yes > p_no) {
    Serial.print("person ("); Serial.print(p_yes * 100.0f, 1); Serial.println("%)");
    return 1;
  } else {
    Serial.print("no person ("); Serial.print(p_no * 100.0f, 1); Serial.println("%)");
    return 0;
  }
}

// =============================================================================
// Audio Feedback
// =============================================================================
void playBeepPattern(int idx) {
  if (idx < 0 || idx >= 2) return;
  const int* p = BEEP_PATTERNS[idx];
  int i = 0;
  while (i < 10 && p[i] != 0) {
    tone(BUZZER_PIN, p[i], p[i+1]);
    delay(p[i+1]);
    if (i+2 < 10 && p[i+2] > 0) { noTone(BUZZER_PIN); delay(p[i+2]); }
    i += 3;
  }
  noTone(BUZZER_PIN);
}

// =============================================================================
// Utilities
// =============================================================================
void printSeparator() {
  Serial.println("----------------------------------------");
}

// =============================================================================
// Live Preview JPEG stream
// Protocol: 0xFF 0xAA <length:4 bytes LE> <JPEG bytes> 0xFF 0xBB
// =============================================================================
void streamPreviewJPEG() {
  myCAM.flush_fifo();
  myCAM.clear_fifo_flag();
  myCAM.start_capture();

  unsigned long t = millis();
  while (!myCAM.get_bit(ARDUCHIP_TRIG, CAP_DONE_MASK)) {
    if (millis() - t > 2000) return;
  }

  uint32_t length = myCAM.read_fifo_length();
  if (length == 0 || length >= 0x7FFFF) return;

  Serial.write((uint8_t)0xFF);
  Serial.write((uint8_t)0xAA);
  Serial.write((uint8_t)(length & 0xFF));
  Serial.write((uint8_t)((length >> 8)  & 0xFF));
  Serial.write((uint8_t)((length >> 16) & 0xFF));
  Serial.write((uint8_t)((length >> 24) & 0xFF));

  myCAM.CS_LOW();
  myCAM.set_fifo_burst();
  for (uint32_t i = 0; i < length; i++) {
    Serial.write(SPI.transfer(0x00));
  }
  myCAM.CS_HIGH();

  Serial.write((uint8_t)0xFF);
  Serial.write((uint8_t)0xBB);
  Serial.flush();
}
