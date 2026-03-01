#!/usr/bin/env python3
"""
Live Camera Preview Server for ArduCAM + Arduino Nano 33 BLE Sense
===================================================================
Reads JPEG frames streamed from the Arduino over Serial and serves
them as an MJPEG HTTP stream that VS Code Simple Browser can display.

Usage:
    python3 preview_server.py

Then in VS Code:
    Ctrl+Shift+P  →  "Simple Browser: Show"  →  http://localhost:8080

The page auto-refreshes and shows:
  - Live camera feed (left)
  - Last inference result (right)
"""

import serial
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

# ─── Config ───────────────────────────────────────────────────────────────────
SERIAL_PORT = '/dev/cu.usbmodem11201'
BAUD_RATE   = 115200
HTTP_PORT   = 8080

FRAME_START = bytes([0xFF, 0xAA])
FRAME_END   = bytes([0xFF, 0xBB])
# ──────────────────────────────────────────────────────────────────────────────

# Shared state (updated by serial thread, read by HTTP thread)
latest_jpeg   = None
latest_result = "Waiting for first frame..."
lock          = threading.Lock()


# ─── Serial Reader Thread ─────────────────────────────────────────────────────
def serial_reader():
    global latest_jpeg, latest_result

    print(f"Opening serial port {SERIAL_PORT} ...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
        ser.dtr = True
    except serial.SerialException as e:
        print(f"ERROR: Cannot open serial port: {e}")
        return

    print("Serial connected. Waiting for frames...")
    buf = bytearray()

    while True:
        try:
            chunk = ser.read(256)
            if not chunk:
                continue
            buf.extend(chunk)

            # Extract text lines (for inference results)
            while b'\n' in buf:
                nl = buf.index(b'\n')
                line = buf[:nl].decode('utf-8', errors='replace').strip()
                buf = buf[nl+1:]
                if line.startswith('RESULT:'):
                    with lock:
                        latest_result = line.replace('RESULT:', '').strip()
                    print(f"  {line}")

            # Extract JPEG frames
            while True:
                start = buf.find(FRAME_START)
                if start == -1:
                    break
                if len(buf) < start + 6:
                    break  # Need 2 marker + 4 length bytes

                length = int.from_bytes(buf[start+2:start+6], 'little')
                end_needed = start + 6 + length + 2

                if len(buf) < end_needed:
                    break  # Frame not fully received yet

                end_marker = buf[start+6+length : start+6+length+2]
                if end_marker == FRAME_END:
                    jpeg_data = bytes(buf[start+6 : start+6+length])
                    with lock:
                        latest_jpeg = jpeg_data
                    buf = buf[end_needed:]
                else:
                    # Bad frame, skip one byte and retry
                    buf = buf[start+1:]

            # Prevent buffer growing indefinitely (keep last 200KB)
            if len(buf) > 200_000:
                buf = buf[-100_000:]

        except Exception as e:
            print(f"Serial error: {e}")
            time.sleep(1)


# ─── HTTP Server ──────────────────────────────────────────────────────────────
HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>ArduCAM Live Preview</title>
  <style>
    body  { background:#111; color:#eee; font-family:monospace; margin:0;
            display:flex; flex-direction:column; align-items:center; padding:20px; }
    h1    { color:#4fc; margin-bottom:10px; font-size:1.2em; }
    .row  { display:flex; gap:30px; align-items:flex-start; }
    .cam  { display:flex; flex-direction:column; align-items:center; }
    img   { border:2px solid #4fc; image-rendering:pixelated;
            width:480px; height:auto; }
    .info { background:#222; border:1px solid #4fc; border-radius:6px;
            padding:20px; min-width:280px; }
    .result { font-size:1.5em; color:#4fc; margin-top:10px; word-break:break-word; }
    .sub  { font-size:0.8em; color:#888; margin-top:20px; }
    .dot  { display:inline-block; width:10px; height:10px; border-radius:50%;
            background:#4fc; animation:pulse 1s infinite; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
  </style>
</head>
<body>
  <h1>TinyML Face Attribute Detection — Live Preview</h1>
  <div class="row">
    <div class="cam">
      <img id="cam" src="/frame.jpg" alt="Camera feed">
      <div class="sub"><span class="dot"></span> Live from ArduCAM OV2640</div>
    </div>
    <div class="info">
      <div style="color:#888;font-size:.9em">INFERENCE RESULT</div>
      <div class="result" id="res">---</div>
      <div class="sub" id="ts">---</div>
    </div>
  </div>

  <script>
    function refresh() {
      // Refresh camera frame
      document.getElementById('cam').src = '/frame.jpg?t=' + Date.now();
      // Fetch result text
      fetch('/result').then(r=>r.text()).then(t=>{
        document.getElementById('res').textContent = t;
        document.getElementById('ts').textContent = new Date().toLocaleTimeString();
      });
    }
    setInterval(refresh, 600);   // refresh ~1.6 fps
  </script>
</body>
</html>"""


class PreviewHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path.startswith('/?'):
            self._send(200, 'text/html', HTML_PAGE.encode())

        elif self.path.startswith('/frame.jpg'):
            with lock:
                data = latest_jpeg
            if data:
                self._send(200, 'image/jpeg', data)
            else:
                # Grey placeholder while waiting
                self._send(200, 'image/jpeg', self._placeholder())

        elif self.path == '/result':
            with lock:
                res = latest_result
            self._send(200, 'text/plain', res.encode())

        else:
            self._send(404, 'text/plain', b'Not found')

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-cache, no-store')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def _placeholder(self):
        """1x1 grey JPEG as placeholder before first frame arrives."""
        return bytes([
            0xff,0xd8,0xff,0xe0,0x00,0x10,0x4a,0x46,0x49,0x46,0x00,0x01,
            0x01,0x00,0x00,0x01,0x00,0x01,0x00,0x00,0xff,0xdb,0x00,0x43,
            0x00,0x08,0x06,0x06,0x07,0x06,0x05,0x08,0x07,0x07,0x07,0x09,
            0x09,0x08,0x0a,0x0c,0x14,0x0d,0x0c,0x0b,0x0b,0x0c,0x19,0x12,
            0x13,0x0f,0x14,0x1d,0x1a,0x1f,0x1e,0x1d,0x1a,0x1c,0x1c,0x20,
            0x24,0x2e,0x27,0x20,0x22,0x2c,0x23,0x1c,0x1c,0x28,0x37,0x29,
            0x2c,0x30,0x31,0x34,0x34,0x34,0x1f,0x27,0x39,0x3d,0x38,0x32,
            0x3c,0x2e,0x33,0x34,0x32,0xff,0xc0,0x00,0x0b,0x08,0x00,0x01,
            0x00,0x01,0x01,0x01,0x11,0x00,0xff,0xc4,0x00,0x1f,0x00,0x00,
            0x01,0x05,0x01,0x01,0x01,0x01,0x01,0x01,0x00,0x00,0x00,0x00,
            0x00,0x00,0x00,0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,
            0x09,0x0a,0x0b,0xff,0xc4,0x00,0xb5,0x10,0x00,0x02,0x01,0x03,
            0x03,0x02,0x04,0x03,0x05,0x05,0x04,0x04,0x00,0x00,0x01,0x7d,
            0x01,0x02,0x03,0x00,0x04,0x11,0x05,0x12,0x21,0x31,0x41,0x06,
            0x13,0x51,0x61,0x07,0x22,0x71,0x14,0x32,0x81,0x91,0xa1,0x08,
            0x23,0x42,0xb1,0xc1,0x15,0x52,0xd1,0xf0,0x24,0x33,0x62,0x72,
            0x82,0x09,0x0a,0x16,0x17,0x18,0x19,0x1a,0x25,0x26,0x27,0x28,
            0x29,0x2a,0x34,0x35,0x36,0x37,0x38,0x39,0x3a,0x43,0x44,0x45,
            0x46,0x47,0x48,0x49,0x4a,0x53,0x54,0x55,0x56,0x57,0x58,0x59,
            0x5a,0x63,0x64,0x65,0x66,0x67,0x68,0x69,0x6a,0x73,0x74,0x75,
            0x76,0x77,0x78,0x79,0x7a,0x83,0x84,0x85,0x86,0x87,0x88,0x89,
            0x8a,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9a,0xa2,0xa3,0xa4,
            0xa5,0xa6,0xa7,0xa8,0xa9,0xaa,0xb2,0xb3,0xb4,0xb5,0xb6,0xb7,
            0xb8,0xb9,0xba,0xc2,0xc3,0xc4,0xc5,0xc6,0xc7,0xc8,0xc9,0xca,
            0xd2,0xd3,0xd4,0xd5,0xd6,0xd7,0xd8,0xd9,0xda,0xe1,0xe2,0xe3,
            0xe4,0xe5,0xe6,0xe7,0xe8,0xe9,0xea,0xf1,0xf2,0xf3,0xf4,0xf5,
            0xf6,0xf7,0xf8,0xf9,0xfa,0xff,0xda,0x00,0x08,0x01,0x01,0x00,
            0x00,0x3f,0x00,0xf5,0x0a,0x28,0xa2,0xff,0xd9
        ])

    def log_message(self, *args):
        pass  # suppress request logs


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Start serial reader in background
    t = threading.Thread(target=serial_reader, daemon=True)
    t.start()

    # Start HTTP server
    server = HTTPServer(('localhost', HTTP_PORT), PreviewHandler)
    print(f"\n{'='*55}")
    print(f"  ArduCAM Live Preview Server")
    print(f"{'='*55}")
    print(f"  HTTP server: http://localhost:{HTTP_PORT}")
    print(f"\n  In VS Code:  Ctrl+Shift+P")
    print(f"               → 'Simple Browser: Show'")
    print(f"               → http://localhost:{HTTP_PORT}")
    print(f"\n  Press Ctrl+C to stop")
    print(f"{'='*55}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
