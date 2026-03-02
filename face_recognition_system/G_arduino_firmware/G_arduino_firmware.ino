/*
 * G. Arduino Firmware — Two-Stage Face Attribute Detection
 *
 * Target Hardware:
 *   - Arduino Nano 33 BLE Sense Rev2
 *   - ArduCam Mini 2MP OV2640 with 8MB FIFO
 *   - Piezo buzzer on pin D6  (optional, for audio feedback)
 *
 * Pipeline:
 *   Stage A: Person Detection  — binary: person vs no_person
 *   Stage B: Attribute Detection — 6 states:
 *               happy | sad | grief | hair | no_hair | multiple
 *
 * Flow:
 *   1. Capture 96x96 grayscale image from ArduCam
 *   2. Run Stage A  →  no person? → Serial + low beep, stop
 *   3. Run Stage B  →  detect attribute → Serial text + buzzer pattern
 *
 * Serial feedback format (115200 baud):
 *   RESULT: HAPPY (95.2%)
 *   RESULT: NO PERSON DETECTED
 *   RESULT: UNKNOWN STATE (low confidence)
 *
 * Buzzer patterns:
 *   No person   — 1 short low beep  (200 Hz)
 *   Person seen — 2 short high beeps (gate beep before attribute)
 *   Happy       — 1 high sustained  (1500 Hz)
 *   Sad         — 1 low  sustained  (400 Hz)
 *   Grief       — 2 low beeps       (400 Hz)
 *   Hair        — 1 mid beep        (900 Hz)
 *   No hair     — 1 short beep      (600 Hz)
 *   Multiple    — 3 rising beeps    (800 → 1000 → 1200 Hz)
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
#include "stage_b_model.h"

// =============================================================================
// Configuration
// =============================================================================
#define CS_PIN              7       // ArduCam chip-select
#define BUZZER_PIN          6       // Piezo buzzer (optional)

#define IMG_WIDTH           96
#define IMG_HEIGHT          96
#define IMG_CHANNELS        1

// Separate arenas: Stage A needs ~46 KB, Stage B needs ~54 KB.
// AllocateTensors() is called ONCE per interpreter in setup; never in loop.
#define ARENA_A_SIZE  (48 * 1024)   // 48 KB for Stage A
#define ARENA_B_SIZE  (56 * 1024)   // 56 KB for Stage B
#define INFERENCE_INTERVAL_MS  500         // ms between frames

// Stage B: number of attribute classes
#define NUM_ATTRIBUTES      6

// Confidence threshold below which we report "unknown state"
#define UNKNOWN_THRESHOLD   0.70f

// =============================================================================
// Attribute labels and buzzer-pattern mapping
// =============================================================================
// Stage B class indices must match the order in processed/metadata.json:
//   0=grief  1=hair  2=happy  3=multiple  4=no_hair  5=sad

const char* ATTRIBUTE_NAMES[NUM_ATTRIBUTES] = {
  "GRIEF",       // class 0
  "HAS HAIR",    // class 1
  "HAPPY",       // class 2
  "MULTIPLE PERSONS", // class 3
  "NO HAIR",     // class 4
  "SAD"          // class 5
};

// Maps Stage B class index → BEEP_PATTERNS index
const int ATTR_TO_PATTERN[NUM_ATTRIBUTES] = {
  5,  // grief    → PATTERN_GRIEF
  6,  // hair     → PATTERN_HAIR
  3,  // happy    → PATTERN_SMILE
  2,  // multiple → PATTERN_MANY
  7,  // no_hair  → PATTERN_NO_HAIR
  4   // sad      → PATTERN_SAD
};

// =============================================================================
// Buzzer patterns  {freq, duration_ms, pause_ms,  ...}  — zero-terminated
// =============================================================================
const int BEEP_PATTERNS[][10] = {
  // 0: No person   — single low beep
  {200,  100, 0,    0,    0,  0,    0,   0, 0, 0},
  // 1: Person seen — two short high beeps (gate tone)
  {1000, 100, 50, 1000, 100,  0,    0,   0, 0, 0},
  // 2: Multiple    — three rising beeps
  {800,   80, 40, 1000,  80, 40, 1200,  80, 0, 0},
  // 3: Smile/Happy — high sustained
  {1500, 180, 0,    0,    0,  0,    0,   0, 0, 0},
  // 4: Sad         — low sustained
  {400,  180, 0,    0,    0,  0,    0,   0, 0, 0},
  // 5: Grief       — two low beeps
  {400,   80, 40,  400,  80,  0,    0,   0, 0, 0},
  // 6: Hair        — mid beep
  {900,  120, 0,    0,    0,  0,    0,   0, 0, 0},
  // 7: No hair     — short quick beep
  {600,   60, 0,    0,    0,  0,    0,   0, 0, 0}
};

#define PATTERN_NO_PERSON   0
#define PATTERN_PERSON      1
#define PATTERN_MANY        2
#define PATTERN_SMILE       3
#define PATTERN_SAD         4
#define PATTERN_GRIEF       5
#define PATTERN_HAIR        6
#define PATTERN_NO_HAIR     7

// =============================================================================
// Global Variables
// =============================================================================
ArduCAM myCAM(OV2640, CS_PIN);

// Camera available flag (set false if camera init fails)
bool camera_ok = false;

// Stage A model pointer (arena allocated on demand)
const tflite::Model* model_stage_a    = nullptr;
const tflite::Model* model_stage_b    = nullptr;

// Pointers to stage interpreters (static storage inside setupTFLite)
tflite::MicroInterpreter* interp_a    = nullptr;
tflite::MicroInterpreter* interp_b    = nullptr;

// Dedicated arenas — each model gets its own, allocated ONCE in setupTFLite()
alignas(16) uint8_t tensor_arena_a[ARENA_A_SIZE];
alignas(16) uint8_t tensor_arena_b[ARENA_B_SIZE];

// Grayscale image buffer
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
int  runStageB(float* confidence);
void playBeepPattern(int pattern_index);
void printSeparator();
void streamPreviewJPEG();

// =============================================================================
// Setup
// =============================================================================
void setup() {
  Serial.begin(115200);
  while (!Serial);   // Wait for host to open serial port before printing

  Serial.println("\n========================================");
  Serial.println("TinyML Face Attribute Detection System");
  Serial.println("Arduino Nano 33 BLE Sense Rev2");
  Serial.println("========================================");

  // Startup chime
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);
  tone(BUZZER_PIN, 1000, 100); delay(150);
  tone(BUZZER_PIN, 1200, 100); delay(150);
  tone(BUZZER_PIN, 1500, 150); delay(200);

  Wire.begin();
  SPI.begin();

  Serial.println("Initialising camera …");
  setupCamera();

  Serial.println("Initialising TensorFlow Lite …");
  setupTFLite();

  Serial.println("\nSystem ready — detecting …\n");
}

// =============================================================================
// Main Loop
// =============================================================================
void loop() {
  unsigned long now = millis();
  if (now - last_inference_time >= INFERENCE_INTERVAL_MS) {
    last_inference_time = now;
    runInference();
    if (camera_ok) streamPreviewJPEG();  // stream live JPEG for VS Code preview
  }
}

// =============================================================================
// Inference Pipeline
// =============================================================================
void runInference() {
  printSeparator();
  unsigned long t0 = millis();

  // --- Capture ---
  if (!captureImage()) {
    Serial.println("ERROR: Camera capture failed");
    playBeepPattern(PATTERN_NO_PERSON);
    return;
  }
  preprocessImage();

  // --- Stage A: Person detection ---
  int stage_a = runStageA();

  if (stage_a <= 0) {
    Serial.println("RESULT: No face detected");
    Serial.println("RECOGNIZED: No face recognized");
    Serial.println("CONFIDENCE: ---");
    playBeepPattern(PATTERN_NO_PERSON);
    printSeparator();
    return;
  }

  // Person confirmed — play gate beep then run Stage B
  playBeepPattern(PATTERN_PERSON);

  // --- Stage B: Attribute detection ---
  float confidence = 0.0f;
  int   attr_class = runStageB(&confidence);

  if (attr_class < 0) {
    Serial.println("RESULT: Face Detected");
    Serial.println("RECOGNIZED: Error");
    Serial.println("CONFIDENCE: ---");
    printSeparator();
    return;
  }

  printSeparator();
  Serial.println("RESULT: Face Detected");

  if (confidence < UNKNOWN_THRESHOLD) {
    Serial.println("RECOGNIZED: Unknown person");
    Serial.print("CONFIDENCE: ");
    Serial.print((int)(confidence * 100.0f));
    Serial.println("%");
  } else {
    const char* label = ATTRIBUTE_NAMES[attr_class];
    Serial.print("RECOGNIZED: ");
    Serial.println(label);
    Serial.print("CONFIDENCE: ");
    Serial.print((int)(confidence * 100.0f));
    Serial.println("%");
    playBeepPattern(ATTR_TO_PATTERN[attr_class]);
  }

  Serial.print("Total inference time: ");
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
    return;  // Non-fatal: continue without camera
  }

  uint8_t vid, pid;
  myCAM.wrSensorReg8_8(0xFF, 0x01);
  myCAM.rdSensorReg8_8(OV2640_CHIPID_HIGH, &vid);
  myCAM.rdSensorReg8_8(OV2640_CHIPID_LOW,  &pid);

  if (vid != 0x26 || (pid != 0x41 && pid != 0x42)) {
    Serial.println("WARNING: OV2640 not detected - camera unavailable");
    Serial.println(">>> Running in TEST MODE with synthetic image <<<");
    camera_ok = false;
    return;  // Non-fatal: continue without camera
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
// TFLite Setup
// Both models share the same tensor_arena sequentially (not simultaneously).
// AllocateTensors() resets the arena before each stage's inference call.
// =============================================================================
void setupTFLite() {
  // AllOpsResolver as function-local static: initialized on first call to
  // setupTFLite() — NOT during global init, so no crash before Serial.begin().
  static tflite::AllOpsResolver resolver;

  // --- Stage A ---
  model_stage_a = tflite::GetModel(stage_a_model);
  if (model_stage_a->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERROR: Stage A schema mismatch!"); while (1);
  }
  Serial.print("Stage A model: ");
  Serial.print(stage_a_model_len);
  Serial.println(" bytes in flash");

  // --- Stage B ---
  model_stage_b = tflite::GetModel(stage_b_model);
  if (model_stage_b->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERROR: Stage B schema mismatch!"); while (1);
  }
  Serial.print("Stage B model: ");
  Serial.print(stage_b_model_len);
  Serial.println(" bytes in flash");

  // Stage A — dedicated 48 KB arena, allocated once here, never again
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

  // Stage B — dedicated 56 KB arena, allocated once here, never again
  static tflite::MicroInterpreter si_b(
    model_stage_b, resolver, tensor_arena_b, ARENA_B_SIZE);
  interp_b = &si_b;
  if (interp_b->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERROR: Stage B tensor allocation failed!"); while (1);
  }
  Serial.print("Stage B arena used: ");
  Serial.print(interp_b->arena_used_bytes());
  Serial.print(" / ");
  Serial.print(ARENA_B_SIZE);
  Serial.println(" bytes");

  Serial.print("Total arenas: ");
  Serial.print((ARENA_A_SIZE + ARENA_B_SIZE) / 1024);
  Serial.println(" KB");
}

// =============================================================================
// Image Capture (or synthetic test image when camera unavailable)
// =============================================================================
// Switch camera back to JPEG 320x240 (used after inference capture)
void restoreJPEGMode() {
  myCAM.set_format(JPEG);
  myCAM.InitCAM();
  myCAM.OV2640_set_JPEG_size(OV2640_320x240);
  delay(100);
  myCAM.clear_fifo_flag();
}

bool captureImage() {
  if (!camera_ok) {
    // Synthetic mid-gray test image (128 = neutral face-like brightness)
    memset(image_buffer, 128, IMG_WIDTH * IMG_HEIGHT);
    return true;
  }

  // Switch to BMP (RGB565) mode so the FIFO contains raw pixels, not JPEG bytes.
  // The TFLite model needs real grayscale pixel values — JPEG bytes would give garbage.
  myCAM.set_format(BMP);
  myCAM.InitCAM();
  myCAM.OV2640_set_JPEG_size(OV2640_160x120);  // smallest available resolution
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

  // Read 160x120 RGB565 (2 bytes/pixel), center-crop to 96x96, convert to grayscale.
  // Crop offsets: skip 32 cols on each side, skip 12 rows top/bottom.
  const int SRC_W    = 160, SRC_H = 120;
  const int COL_SKIP = (SRC_W - IMG_WIDTH)  / 2;  // 32
  const int ROW_SKIP = (SRC_H - IMG_HEIGHT) / 2;  // 12

  myCAM.CS_LOW();
  myCAM.set_fifo_burst();

  int out = 0;
  for (int r = 0; r < SRC_H; r++) {
    bool in_row = (r >= ROW_SKIP && r < ROW_SKIP + IMG_HEIGHT);
    for (int c = 0; c < SRC_W; c++) {
      uint8_t hi = SPI.transfer(0x00);
      uint8_t lo = SPI.transfer(0x00);
      if (in_row && c >= COL_SKIP && c < COL_SKIP + IMG_WIDTH) {
        // RGB565 -> 8-bit grayscale  (luminance weights: R*0.299, G*0.587, B*0.114)
        uint8_t r8 = (hi & 0xF8);                          // R (5-bit -> 8-bit, lower 3 = 0)
        uint8_t g8 = ((hi & 0x07) << 5) | ((lo & 0xE0) >> 3); // G (6-bit -> 8-bit)
        uint8_t b8 = (lo & 0x1F) << 3;                    // B (5-bit -> 8-bit)
        image_buffer[out++] = (uint8_t)((77u * r8 + 150u * g8 + 29u * b8) >> 8);
      }
    }
  }

  myCAM.CS_HIGH();
  restoreJPEGMode();  // switch back to JPEG for streamPreviewJPEG()
  return (out == IMG_WIDTH * IMG_HEIGHT);
}

// =============================================================================
// Preprocessing — clamp pixel values (already uint8)
// =============================================================================
void preprocessImage() {
  for (int i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++) {
    if (image_buffer[i] > 255) image_buffer[i] = 255;
  }
}

// =============================================================================
// Helper: copy image_buffer into a TFLite INT8 or float input tensor
// =============================================================================
static void fillInputTensor(TfLiteTensor* input) {
  if (input->type == kTfLiteInt8) {
    float  scale      = input->params.scale;
    int32_t zero_pt   = input->params.zero_point;
    int8_t* data      = input->data.int8;
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
// Returns 1 (person) or 0 (no person) or -1 (error)
// NOTE: Calls AllocateTensors() to claim the shared arena for Stage A.
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
    float s = output->params.scale;
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
// Stage B — Attribute detection
// Returns class index [0..NUM_ATTRIBUTES-1], writes max confidence to *conf
// NOTE: Calls AllocateTensors() to claim the shared arena for Stage B.
// =============================================================================
int runStageB(float* conf) {
  TfLiteTensor* input  = interp_b->input(0);
  TfLiteTensor* output = interp_b->output(0);

  fillInputTensor(input);

  unsigned long t = millis();
  if (interp_b->Invoke() != kTfLiteOk) {
    Serial.println("Stage B invoke failed"); *conf = 0; return -1;
  }
  Serial.print("Stage B: ");
  Serial.print(millis() - t);
  Serial.println(" ms");

  int   best_idx  = 0;
  float best_prob = -1e9f;

  for (int i = 0; i < NUM_ATTRIBUTES; i++) {
    float prob;
    if (output->type == kTfLiteInt8) {
      float s  = output->params.scale;
      int32_t z = output->params.zero_point;
      prob = (output->data.int8[i] - z) * s;
    } else {
      prob = output->data.f[i];
    }
    if (prob > best_prob) { best_prob = prob; best_idx = i; }
  }

  *conf = best_prob;
  return best_idx;
}

// =============================================================================
// Audio Feedback
// =============================================================================
void playBeepPattern(int idx) {
  if (idx < 0 || idx >= 8) return;
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
// Live Preview: capture a fresh JPEG and stream it over Serial.
// Protocol: 0xFF 0xAA <length:4 bytes LE> <JPEG bytes> 0xFF 0xBB
// The Python preview_server.py listens for this and serves MJPEG on :8080
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

  // Frame start marker
  Serial.write((uint8_t)0xFF);
  Serial.write((uint8_t)0xAA);
  // Length (little-endian 4 bytes)
  Serial.write((uint8_t)(length & 0xFF));
  Serial.write((uint8_t)((length >> 8) & 0xFF));
  Serial.write((uint8_t)((length >> 16) & 0xFF));
  Serial.write((uint8_t)((length >> 24) & 0xFF));

  // Stream JPEG bytes directly from FIFO to Serial
  myCAM.CS_LOW();
  myCAM.set_fifo_burst();
  for (uint32_t i = 0; i < length; i++) {
    Serial.write(SPI.transfer(0x00));
  }
  myCAM.CS_HIGH();

  // Frame end marker
  Serial.write((uint8_t)0xFF);
  Serial.write((uint8_t)0xBB);
  Serial.flush();
}
