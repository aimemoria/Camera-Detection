/*
 * Minimal diagnostic sketch — no camera, no inference
 * Just proves serial works and tests TFLite allocation.
 */

#include <Wire.h>
#include <SPI.h>

#ifdef swap
#undef swap
#endif

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "../stage_a_model.h"
#include "../stage_b_model.h"

#define ARENA_SIZE (100 * 1024)
alignas(16) static uint8_t tensor_arena[ARENA_SIZE];

void setup() {
  Serial.begin(115200);
  delay(3000);   // give USB host time to open port

  Serial.println("=== BOOT OK ===");
  Serial.flush();

  // Test Stage A allocation
  const tflite::Model* model_a = tflite::GetModel(stage_a_model);
  if (model_a->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Stage A: schema mismatch"); Serial.flush(); while(1);
  }
  Serial.println("Stage A model: schema OK");

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter interp_a(
    model_a, resolver, tensor_arena, ARENA_SIZE);

  if (interp_a.AllocateTensors() != kTfLiteOk) {
    Serial.println("Stage A: AllocateTensors FAILED"); Serial.flush(); while(1);
  }
  Serial.print("Stage A tensors allocated, arena used: ");
  Serial.println(interp_a.arena_used_bytes());
  Serial.flush();

  // Test Stage B — note: shares same arena (will conflict!)
  const tflite::Model* model_b = tflite::GetModel(stage_b_model);
  if (model_b->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Stage B: schema mismatch"); Serial.flush(); while(1);
  }
  Serial.println("Stage B model: schema OK");

  static tflite::MicroInterpreter interp_b(
    model_b, resolver, tensor_arena, ARENA_SIZE);

  if (interp_b.AllocateTensors() != kTfLiteOk) {
    Serial.println("Stage B: AllocateTensors FAILED (shared arena conflict)");
    Serial.flush();
    // Don't halt — report and continue
  } else {
    Serial.print("Stage B tensors allocated, arena used: ");
    Serial.println(interp_b.arena_used_bytes());
  }
  Serial.flush();

  Serial.println("=== SETUP COMPLETE ===");
  Serial.flush();
}

void loop() {
  Serial.println("ALIVE");
  Serial.flush();
  delay(2000);
}
