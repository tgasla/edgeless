#!/usr/bin/env python3
"""
Simple HTTP server for the Vision Demo.

- Port 8080: Serves images from ./images/ (camera source reads from here)
- Port 8081: Accepts POST requests and saves image files to ./vision_output/

Usage:
    # Terminal 1 - Serve source images:
    cd examples/vision_demo && python3 -m http.server 8080 --directory images

    # Terminal 2 - Save output frames:
    cd examples/vision_demo && python3 file_save_server.py
"""

import os
import http.server
import socketserver

OUTPUT_DIR = "./vision_output"
PORT = 8081


class FileSaveHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        # Extract filename from path (e.g., /save/frame_0001.png -> frame_0001.png)
        path = self.path.strip("/")
        if path.startswith("save/"):
            filename = path[5:]  # Remove "save/" prefix
        else:
            filename = path

        # Read the body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        # Save to disk
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(body)

        print(f"Saved {filepath} ({len(body)} bytes)")

        # Send 200 OK
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(f"Saved {filename}".encode())

    def log_message(self, format, *args):
        print(f"[FileServer] {args[0]}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with socketserver.TCPServer(("", PORT), FileSaveHandler) as httpd:
        print(f"File Save Server listening on port {PORT}")
        print(f"Saving files to {os.path.abspath(OUTPUT_DIR)}")
        httpd.serve_forever()
