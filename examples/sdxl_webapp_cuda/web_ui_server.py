import os, json, uuid, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.request

PORT = 8080
PENDING_TASKS = {}

HTML_UI = """
<!DOCTYPE html>
<html>
<head><title>Edgeless AI Generator</title><style>body{font-family:sans-serif; padding:20px;} .container{max-width:600px; margin:auto;} input, button{width:100%; margin-top:10px; padding:10px;}</style></head>
<body>
<div class="container">
    <h2>Edgeless Image-to-Image</h2>
    <input type="file" id="imageInput" accept="image/png, image/jpeg">
    <input type="text" id="promptInput" placeholder="Enter your prompt..." value="A beautiful cinematic fantasy landscape, 4k">
    <label>Creativity (Timesteps): <span id="tempVal">350</span></label>
    <input type="range" id="tempInput" min="100" max="600" value="350" oninput="document.getElementById('tempVal').innerText=this.value">
    <button onclick="generate()">Generate ✨</button>
    <div id="status" style="margin-top:10px; font-weight:bold;"></div>
    <img id="resultImg" style="width:100%; margin-top:20px; border-radius:10px; display:none;">
</div>
<script>
async function generate() {
    const file = document.getElementById('imageInput').files[0];
    if (!file) return alert("Upload an image first!");
    
    document.getElementById('status').innerText = "Processing in Edgeless Cluster...";
    document.getElementById('resultImg').style.display = "none";
    
    const reader = new FileReader();
    reader.onload = async function() {
        const base64 = reader.result.split(',')[1]; // Strip data header
        const payload = {
            prompt: document.getElementById('promptInput').value,
            timestep: parseInt(document.getElementById('tempInput').value),
            image_base64: base64
        };
        
        const res = await fetch('/generate', { method: 'POST', body: JSON.stringify(payload) });
        const data = await res.json();
        
        document.getElementById('resultImg').src = "data:image/png;base64," + data.image_base64;
        document.getElementById('resultImg').style.display = "block";
        document.getElementById('status').innerText = "Done!";
    };
    reader.readAsDataURL(file);
}
</script>
</body>
</html>
"""

class BridgeHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_UI.encode())

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(length))

        # 1. Browser asks for generation
        if self.path == '/generate':
            req_id = str(uuid.uuid4())
            PENDING_TASKS[req_id] = None
            
            # Forward into Edgeless http-ingress
            edgeless_payload = {"id": req_id, **body}
            req = urllib.request.Request("http://127.0.0.1:7007/", data=json.dumps(edgeless_payload).encode(), headers={'Content-Type': 'application/json'})
            urllib.request.urlopen(req)

            # Wait for Edgeless to hit the webhook
            while PENDING_TASKS[req_id] is None:
                time.sleep(0.1)

            result_b64 = PENDING_TASKS.pop(req_id)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"image_base64": result_b64}).encode())

        # 2. Edgeless visualizer sends the result back
        elif self.path == '/webhook':
            req_id = body.get("id")
            if req_id in PENDING_TASKS:
                PENDING_TASKS[req_id] = body.get("image_base64")
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")

if __name__ == "__main__":
    ThreadingHTTPServer(("", PORT), BridgeHandler).serve_forever()