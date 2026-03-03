/*
 * Camera-only test — no TFLite, no inference.
 * Captures JPEG from ArduCAM OV2640 and streams it over Serial
 * so preview_server.py can display it at http://localhost:8080
 *
 * Use this first to confirm the camera is wired correctly and
 * the live preview works before running the full firmware.
 */

#include <Wire.h>
#include <SPI.h>
#include <ArduCAM.h>
#include "memorysaver.h"

#define CS_PIN   7

ArduCAM myCAM(OV2640, CS_PIN);

// Same binary protocol as preview_server.py expects:
// 0xFF 0xAA  <length: 4 bytes LE>  <JPEG bytes>  0xFF 0xBB
void streamJPEG() {
  myCAM.flush_fifo();
  myCAM.clear_fifo_flag();
  myCAM.start_capture();

  unsigned long t = millis();
  while (!myCAM.get_bit(ARDUCHIP_TRIG, CAP_DONE_MASK)) {
    if (millis() - t > 3000) {
      Serial.println("Capture timeout");
      return;
    }
  }

  uint32_t length = myCAM.read_fifo_length();
  if (length == 0 || length >= 0x7FFFF) {
    Serial.println("Bad FIFO length");
    return;
  }

  // Frame start marker + 4-byte length
  Serial.write((uint8_t)0xFF);
  Serial.write((uint8_t)0xAA);
  Serial.write((uint8_t)(length & 0xFF));
  Serial.write((uint8_t)((length >> 8) & 0xFF));
  Serial.write((uint8_t)((length >> 16) & 0xFF));
  Serial.write((uint8_t)((length >> 24) & 0xFF));

  // Stream JPEG bytes from FIFO
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

void setup() {
  Serial.begin(115200);
  while (!Serial);

  Serial.println("=== Camera Test ===");

  Wire.begin();
  SPI.begin();

  pinMode(CS_PIN, OUTPUT);
  digitalWrite(CS_PIN, HIGH);

  // SPI test
  myCAM.write_reg(ARDUCHIP_TEST1, 0x55);
  if (myCAM.read_reg(ARDUCHIP_TEST1) != 0x55) {
    Serial.println("ERROR: SPI test failed — check wiring on D7/D11/D12/D13");
    while (1);
  }
  Serial.println("SPI: OK");

  // Sensor ID check
  uint8_t vid, pid;
  myCAM.wrSensorReg8_8(0xFF, 0x01);
  myCAM.rdSensorReg8_8(OV2640_CHIPID_HIGH, &vid);
  myCAM.rdSensorReg8_8(OV2640_CHIPID_LOW,  &pid);

  if (vid != 0x26 || (pid != 0x41 && pid != 0x42)) {
    Serial.print("ERROR: OV2640 not found. VID=0x");
    Serial.print(vid, HEX);
    Serial.print(" PID=0x");
    Serial.println(pid, HEX);
    while (1);
  }
  Serial.println("Camera: OV2640 OK");

  myCAM.set_format(JPEG);
  myCAM.InitCAM();
  myCAM.OV2640_set_JPEG_size(OV2640_320x240);
  delay(100);
  myCAM.clear_fifo_flag();

  Serial.println("Streaming... open http://localhost:8080 in VS Code Simple Browser");
}

void loop() {
  streamJPEG();
  delay(100);  // ~10 fps
}
