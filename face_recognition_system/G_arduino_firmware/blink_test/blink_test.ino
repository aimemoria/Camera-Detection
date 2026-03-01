void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(115200);
  while (!Serial);   // Wait for host to open serial port (DTR)
  Serial.println("HELLO FROM NANO 33 BLE");
  Serial.flush();
}
void loop() {
  digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
  Serial.print("ALIVE tick=");
  Serial.println(millis());
  Serial.flush();
  delay(1000);
}
