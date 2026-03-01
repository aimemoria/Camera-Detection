/*
 * TinyML Person Detection - Arduino Nano 33 Sense + ArduCam
 * 
 * This firmware:
 * 1. Captures frames from ArduCam OV7675 camera module
 * 2. Resizes and preprocesses image to match model input
 * 3. Runs inference using TensorFlow Lite for Microcontrollers
 * 4. Outputs prediction confidence via Serial
 * 5. Measures inference latency and FPS
 * 
 * Hardware:
 * - Arduino Nano 33 BLE Sense (256KB RAM, 1MB Flash)
 * - ArduCam OV7675 camera module (I2C + SPI)
 * 
 * Libraries Required:
 * - Arduino_TensorFlowLite (by TensorFlow Authors)
 * - ArduCAM (by Lee)
 * 
 * Install via Arduino Library Manager:
 * 1. Tools → Manage Libraries
 * 2. Search "Arduino_TensorFlowLite" → Install
 * 3. Search "ArduCAM" → Install
 * 
 * Wiring:
 * ArduCam → Arduino Nano 33 Sense
 * --------------------------------
 * CS   → D7 (configurable)
 * MOSI → D11 (SPI MOSI)
 * MISO → D12 (SPI MISO)
 * SCK  → D13 (SPI SCK)
 * SDA  → A4 (I2C SDA)
 * SCL  → A5 (I2C SCL)
 * VCC  → 3.3V
 * GND  → GND
 * 
 * Memory Budget:
 * - Model: ~50-80 KB (Flash)
 * - Tensor arena: 100-150 KB (RAM)
 * - Image buffer: ~28 KB (RAM, 96x96x3)
 * - Total RAM: ~130-180 KB (under 256 KB limit)
 */

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <ArduCAM.h>
#include <Wire.h>
#include <SPI.h>

// Include model data (generated from convert_to_tflite.py)
#include "model_data.h"

// ==================== CONFIGURATION ====================

// Camera configuration
#define CS_PIN 7                    // Chip select pin for ArduCam
#define IMAGE_WIDTH 96              // Model input width
#define IMAGE_HEIGHT 96             // Model input height
#define IMAGE_CHANNELS 3            // RGB channels (or 1 for grayscale)

// Camera capture resolution (will be resized to IMAGE_WIDTH x IMAGE_HEIGHT)
#define CAPTURE_WIDTH 160           // ArduCam native resolution
#define CAPTURE_HEIGHT 120

// Tensor arena size (adjust based on model complexity)
// Rule of thumb: ~100-150 KB for small models
constexpr int kTensorArenaSize = 120 * 1024;  // 120 KB

// Inference threshold
constexpr float kConfidenceThreshold = 0.6;   // 60% confidence

// Class labels
const char* kClassLabels[] = {"no_person", "person"};

// ==================== GLOBAL VARIABLES ====================

// TensorFlow Lite objects
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Tensor arena (memory pool for TFLite operations)
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// ArduCam object
ArduCAM myCAM(OV7675, CS_PIN);

// Image buffer
uint8_t image_buffer[IMAGE_HEIGHT][IMAGE_WIDTH][IMAGE_CHANNELS];

// Performance metrics
unsigned long inference_count = 0;
unsigned long total_inference_time_ms = 0;
unsigned long last_fps_print = 0;
unsigned int frame_count = 0;

// ==================== SETUP ====================

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 5000);  // Wait up to 5 seconds for Serial
  
  Serial.println("====================================");
  Serial.println("TinyML Person Detection");
  Serial.println("Arduino Nano 33 Sense + ArduCam");
  Serial.println("====================================");
  
  // Initialize I2C and SPI
  Wire.begin();
  SPI.begin();
  
  // Initialize camera
  Serial.println("\n[1/5] Initializing ArduCam...");
  if (!initCamera()) {
    Serial.println("ERROR: Camera initialization failed!");
    while (1) {
      delay(1000);
    }
  }
  Serial.println("✓ Camera initialized");
  
  // Set up TensorFlow Lite
  Serial.println("\n[2/5] Setting up TensorFlow Lite...");
  setupTFLite();
  Serial.println("✓ TensorFlow Lite ready");
  
  // Print model info
  Serial.println("\n[3/5] Model Information:");
  printModelInfo();
  
  // Print memory usage
  Serial.println("\n[4/5] Memory Usage:");
  printMemoryUsage();
  
  // Test inference
  Serial.println("\n[5/5] Running test inference...");
  testInference();
  
  Serial.println("\n====================================");
  Serial.println("System ready! Starting detection...");
  Serial.println("====================================\n");
  
  delay(1000);
}

// ==================== MAIN LOOP ====================

void loop() {
  unsigned long loop_start = millis();
  
  // Step 1: Capture frame from camera
  if (!captureFrame()) {
    Serial.println("ERROR: Frame capture failed");
    delay(100);
    return;
  }
  
  // Step 2: Preprocess image
  preprocessImage();
  
  // Step 3: Run inference
  unsigned long inference_start = millis();
  
  if (!runInference()) {
    Serial.println("ERROR: Inference failed");
    delay(100);
    return;
  }
  
  unsigned long inference_time = millis() - inference_start;
  
  // Step 4: Get prediction
  float no_person_score = getOutputScore(0);
  float person_score = getOutputScore(1);
  
  int predicted_class = (person_score > no_person_score) ? 1 : 0;
  float confidence = max(no_person_score, person_score);
  
  // Step 5: Print results
  printPrediction(predicted_class, confidence, inference_time);
  
  // Update metrics
  inference_count++;
  total_inference_time_ms += inference_time;
  frame_count++;
  
  // Print FPS every 5 seconds
  if (millis() - last_fps_print >= 5000) {
    printPerformanceMetrics();
    last_fps_print = millis();
    frame_count = 0;
  }
  
  // Small delay to avoid overwhelming serial output
  delay(100);
}

// ==================== CAMERA FUNCTIONS ====================

bool initCamera() {
  pinMode(CS_PIN, OUTPUT);
  digitalWrite(CS_PIN, HIGH);
  
  // Reset camera
  myCAM.write_reg(0x07, 0x80);
  delay(100);
  myCAM.write_reg(0x07, 0x00);
  delay(100);
  
  // Check camera presence
  myCAM.write_reg(ARDUCHIP_TEST1, 0x55);
  uint8_t test = myCAM.read_reg(ARDUCHIP_TEST1);
  if (test != 0x55) {
    Serial.print("ERROR: Camera not detected (test=0x");
    Serial.print(test, HEX);
    Serial.println(")");
    return false;
  }
  
  // Initialize camera with appropriate format
  myCAM.set_format(JPEG);
  myCAM.InitCAM();
  
  // Set resolution (adjust to your camera's supported resolutions)
  myCAM.OV7675_set_JPEG_size(OV7675_160x120);
  
  delay(1000);
  
  return true;
}

bool captureFrame() {
  // For this example, we'll use a simplified capture approach
  // In production, you'd read JPEG from ArduCam and decode it
  
  // Start capture
  myCAM.flush_fifo();
  myCAM.clear_fifo_flag();
  myCAM.start_capture();
  
  // Wait for capture complete
  unsigned long timeout = millis() + 1000;
  while (!myCAM.get_bit(ARDUCHIP_TRIG, CAP_DONE_MASK)) {
    if (millis() > timeout) {
      Serial.println("ERROR: Capture timeout");
      return false;
    }
    delay(1);
  }
  
  // Read image data from FIFO
  // NOTE: This is simplified. Real implementation requires:
  // 1. Reading JPEG data from FIFO
  // 2. Decoding JPEG to RGB
  // 3. Resizing to model input size
  
  // For demonstration, we'll fill with dummy data
  // In production, replace this with actual camera data
  fillDummyImage();
  
  return true;
}

void fillDummyImage() {
  // Fill with gradient pattern for testing
  // Replace this with actual camera data in production
  for (int y = 0; y < IMAGE_HEIGHT; y++) {
    for (int x = 0; x < IMAGE_WIDTH; x++) {
      image_buffer[y][x][0] = (x * 255) / IMAGE_WIDTH;        // R
      image_buffer[y][x][1] = (y * 255) / IMAGE_HEIGHT;       // G
      image_buffer[y][x][2] = ((x + y) * 255) / (IMAGE_WIDTH + IMAGE_HEIGHT);  // B
    }
  }
}

// ==================== PREPROCESSING ====================

void preprocessImage() {
  // Get input tensor
  TfLiteTensor* input = interpreter->input(0);
  
  // Check if input is INT8 or FLOAT32
  bool is_quantized = (input->type == kTfLiteInt8);
  
  // Get quantization parameters if INT8
  float input_scale = 1.0f;
  int input_zero_point = 0;
  
  if (is_quantized && input->params.scale != 0) {
    input_scale = input->params.scale;
    input_zero_point = input->params.zero_point;
  }
  
  // Copy and normalize image to input tensor
  for (int y = 0; y < IMAGE_HEIGHT; y++) {
    for (int x = 0; x < IMAGE_WIDTH; x++) {
      for (int c = 0; c < IMAGE_CHANNELS; c++) {
        int index = (y * IMAGE_WIDTH + x) * IMAGE_CHANNELS + c;
        
        // Normalize to [-1, 1] range
        float normalized = (image_buffer[y][x][c] / 127.5f) - 1.0f;
        
        if (is_quantized) {
          // Quantize to INT8
          int8_t quantized = (int8_t)((normalized / input_scale) + input_zero_point);
          input->data.int8[index] = quantized;
        } else {
          // Keep as FLOAT32
          input->data.f[index] = normalized;
        }
      }
    }
  }
}

// ==================== TENSORFLOW LITE FUNCTIONS ====================

void setupTFLite() {
  // Set up logging
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  
  // Load model
  model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("ERROR: Model schema version mismatch. Expected ");
    Serial.print(TFLITE_SCHEMA_VERSION);
    Serial.print(", got ");
    Serial.println(model->version());
    while (1);
  }
  
  // Set up ops resolver (use AllOpsResolver for compatibility)
  static tflite::AllOpsResolver resolver;
  
  // Build interpreter
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize, error_reporter
  );
  interpreter = &static_interpreter;
  
  // Allocate tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors() failed");
    while (1);
  }
  
  // Get input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);
}

bool runInference() {
  // Invoke interpreter
  TfLiteStatus invoke_status = interpreter->Invoke();
  
  if (invoke_status != kTfLiteOk) {
    Serial.println("ERROR: Invoke() failed");
    return false;
  }
  
  return true;
}

float getOutputScore(int class_index) {
  TfLiteTensor* output = interpreter->output(0);
  
  // Check if output is quantized
  if (output->type == kTfLiteInt8) {
    // Dequantize
    float scale = output->params.scale;
    int zero_point = output->params.zero_point;
    int8_t quantized_value = output->data.int8[class_index];
    float score = (quantized_value - zero_point) * scale;
    
    // Apply softmax approximation if needed
    // For simplicity, we assume output is already softmax-ed
    return score;
  } else {
    // Float32 output
    return output->data.f[class_index];
  }
}

void testInference() {
  // Fill input with random data
  TfLiteTensor* input = interpreter->input(0);
  int input_size = input->bytes;
  
  for (int i = 0; i < input_size; i++) {
    if (input->type == kTfLiteInt8) {
      input->data.int8[i] = random(-128, 127);
    } else {
      input->data.f[i] = random(-100, 100) / 100.0f;
    }
  }
  
  // Run inference
  unsigned long start = millis();
  bool success = runInference();
  unsigned long duration = millis() - start;
  
  if (success) {
    Serial.print("✓ Test inference successful (");
    Serial.print(duration);
    Serial.println(" ms)");
  } else {
    Serial.println("✗ Test inference failed");
  }
}

// ==================== UTILITY FUNCTIONS ====================

void printModelInfo() {
  Serial.print("  Model size: ");
  Serial.print(model_data_len);
  Serial.print(" bytes (");
  Serial.print(model_data_len / 1024.0, 2);
  Serial.println(" KB)");
  
  TfLiteTensor* input = interpreter->input(0);
  Serial.print("  Input shape: ");
  Serial.print(input->dims->data[1]);
  Serial.print(" x ");
  Serial.print(input->dims->data[2]);
  Serial.print(" x ");
  Serial.println(input->dims->data[3]);
  
  Serial.print("  Input type: ");
  Serial.println(input->type == kTfLiteInt8 ? "INT8" : "FLOAT32");
  
  TfLiteTensor* output = interpreter->output(0);
  Serial.print("  Output classes: ");
  Serial.println(output->dims->data[1]);
}

void printMemoryUsage() {
  Serial.print("  Tensor arena: ");
  Serial.print(kTensorArenaSize / 1024);
  Serial.println(" KB");
  
  Serial.print("  Arena used: ");
  Serial.print(interpreter->arena_used_bytes() / 1024);
  Serial.println(" KB");
  
  int image_buffer_size = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS;
  Serial.print("  Image buffer: ");
  Serial.print(image_buffer_size / 1024.0, 2);
  Serial.println(" KB");
  
  int total_ram = kTensorArenaSize + image_buffer_size;
  Serial.print("  Total RAM usage: ");
  Serial.print(total_ram / 1024);
  Serial.println(" KB");
  
  if (total_ram > 256 * 1024) {
    Serial.println("  ⚠️ WARNING: RAM usage exceeds 256 KB!");
  } else {
    Serial.println("  ✓ RAM usage within limits");
  }
}

void printPrediction(int predicted_class, float confidence, unsigned long inference_time) {
  Serial.print("[Frame ");
  Serial.print(inference_count);
  Serial.print("] ");
  
  Serial.print(kClassLabels[predicted_class]);
  Serial.print(" (");
  Serial.print(confidence * 100, 1);
  Serial.print("%) | ");
  
  Serial.print("Inference: ");
  Serial.print(inference_time);
  Serial.print(" ms | ");
  
  float fps = 1000.0 / (float)inference_time;
  Serial.print("FPS: ");
  Serial.println(fps, 1);
}

void printPerformanceMetrics() {
  Serial.println("\n--- Performance Metrics (last 5 seconds) ---");
  
  if (inference_count > 0) {
    float avg_inference_time = (float)total_inference_time_ms / (float)inference_count;
    float avg_fps = 1000.0 / avg_inference_time;
    
    Serial.print("  Frames processed: ");
    Serial.println(frame_count);
    
    Serial.print("  Avg inference time: ");
    Serial.print(avg_inference_time, 2);
    Serial.println(" ms");
    
    Serial.print("  Avg FPS: ");
    Serial.println(avg_fps, 2);
    
    Serial.print("  Total inferences: ");
    Serial.println(inference_count);
  }
  
  Serial.println("-------------------------------------------\n");
}

// ==================== MEMORY OPTIMIZATION TIPS ====================

/*
 * MEMORY OPTIMIZATION BEST PRACTICES:
 * 
 * 1. Use INT8 quantization instead of FLOAT32 (4x size reduction)
 * 2. Reduce tensor arena size if model is smaller
 * 3. Use grayscale instead of RGB (3x reduction)
 * 4. Reduce input resolution (96x96 → 64x64)
 * 5. Use MicroMutableOpResolver instead of AllOpsResolver
 * 6. Disable Serial debugging in production
 * 7. Free unused buffers after initialization
 * 8. Use PROGMEM for model data (store in Flash, not RAM)
 * 
 * Example: Switching to MicroMutableOpResolver
 * 
 * static tflite::MicroMutableOpResolver<5> micro_op_resolver;
 * micro_op_resolver.AddConv2D();
 * micro_op_resolver.AddMaxPool2D();
 * micro_op_resolver.AddFullyConnected();
 * micro_op_resolver.AddReshape();
 * micro_op_resolver.AddSoftmax();
 * 
 * This saves ~20-30 KB compared to AllOpsResolver.
 */
