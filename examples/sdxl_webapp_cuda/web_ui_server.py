import os
import json
import uuid
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.request

PORT = 8080
PENDING_TASKS = {}

HTML_UI = """<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Engine Studio</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #121212;
            color: #ffffff;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 40px 20px;
            min-height: 100vh;
            box-sizing: border-box;
        }

        .container {
            background: #1e1e1e;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.5);
            width: 100%;
            max-width: 600px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        h2 {
            margin: 0 0 10px 0;
            text-align: center;
            font-weight: 600;
        }

        label {
            font-size: 14px;
            color: #aaa;
            margin-bottom: 5px;
            display: block;
        }

        input[type="file"] {
            display: none;
        }

        .upload-area {
            background: #2c2c2c;
            border: 2px dashed #444;
            border-radius: 8px;
            padding: 30px;
            text-align: center;
            cursor: pointer;
            transition: border-color 0.2s, background 0.2s;
        }

        .upload-area:hover {
            border-color: #007bff;
            background: rgba(0, 123, 255, 0.05);
        }

        .upload-area.dragover {
            border-color: #007bff;
            background: rgba(0, 123, 255, 0.1);
        }

        .upload-area input {
            display: none;
        }

        .upload-text {
            color: #aaa;
        }

        .upload-text small {
            display: block;
            margin-top: 5px;
            color: #666;
        }

        #previewImage {
            max-width: 100%;
            max-height: 200px;
            border-radius: 6px;
            display: none;
            margin-top: 10px;
            margin-left: auto;
            margin-right: auto;
        }

        textarea {
            background: #2c2c2c;
            border: 1px solid #444;
            color: white;
            padding: 12px;
            border-radius: 6px;
            font-size: 16px;
            width: 100%;
            box-sizing: border-box;
            resize: vertical;
            font-family: inherit;
        }

        textarea:focus {
            outline: none;
            border-color: #007bff;
        }

        .slider-container {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .slider-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .slider-value {
            background: #2c2c2c;
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 14px;
            color: #007bff;
            font-weight: bold;
        }

        input[type="range"] {
            width: 100%;
            height: 6px;
            border-radius: 3px;
            background: #2c2c2c;
            appearance: none;
            cursor: pointer;
        }

        input[type="range"]::-webkit-slider-thumb {
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #007bff;
            cursor: pointer;
            transition: background 0.2s;
        }

        input[type="range"]::-webkit-slider-thumb:hover {
            background: #0056b3;
        }

        button {
            background: #007bff;
            color: white;
            border: none;
            padding: 15px;
            font-size: 18px;
            font-weight: bold;
            border-radius: 6px;
            cursor: pointer;
            transition: background 0.2s, transform 0.1s;
        }

        button:hover {
            background: #0056b3;
        }

        button:active {
            transform: scale(0.98);
        }

        button:disabled {
            background: #555;
            cursor: not-allowed;
            transform: none;
        }

        #loading {
            display: none;
            text-align: center;
            font-weight: bold;
            color: #007bff;
            padding: 10px;
            background: rgba(0, 123, 255, 0.1);
            border-radius: 6px;
        }

        #loading small {
            display: block;
            margin-top: 5px;
            font-weight: normal;
            color: #888;
        }

        #historyLoading {
            display: none;
            text-align: center;
            font-weight: bold;
            color: #28a745;
            padding: 10px;
            background: rgba(40, 167, 69, 0.1);
            border-radius: 6px;
        }

        #historyLoading small {
            display: block;
            margin-top: 5px;
            font-weight: normal;
            color: #888;
        }

        #resultImage {
            width: 100%;
            border-radius: 8px;
            margin-top: 10px;
            display: none;
            border: 2px dashed #444;
            object-fit: contain;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        /* History Panel Styles */
        #historyPanel {
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #444;
        }

        #historyBtn {
            background: #28a745;
            margin-bottom: 10px;
        }

        #historyBtn:hover {
            background: #218838;
        }

        #historyBtn.loading {
            background: #555;
        }

        #historyContent {
            display: none;
            max-height: 400px;
            overflow-y: auto;
            background: #2c2c2c;
            border-radius: 8px;
            padding: 10px;
        }

        .history-item {
            background: #1e1e1e;
            border-radius: 6px;
            padding: 10px;
            margin-bottom: 10px;
        }

        .history-item:last-child {
            margin-bottom: 0;
        }

        .history-item-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }

        .history-item-prompt {
            font-size: 12px;
            color: #aaa;
            margin: 0;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            max-width: 70%;
            cursor: pointer;
        }

        .history-item-prompt.expanded {
            white-space: normal;
            overflow: visible;
            text-overflow: unset;
            word-break: break-word;
        }

        .history-item-time {
            font-size: 10px;
            color: #666;
        }

        .history-item-meta {
            font-size: 11px;
            color: #888;
            margin-bottom: 8px;
        }

        .history-item-images {
            display: flex;
            gap: 10px;
        }

        .history-item-images img {
            width: 200px;
            height: 200px;
            object-fit: contain;
            border-radius: 4px;
            background: #333;
            cursor: pointer;
            transition: transform 0.2s;
        }

        .history-item-images img:hover {
            transform: scale(1.05);
        }

        .history-empty {
            text-align: center;
            color: #666;
            padding: 20px;
        }

        /* Fullscreen image modal */
        #imageModal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.9);
        }

        #imageModalContent {
            margin: auto;
            display: block;
            max-width: 90%;
            max-height: 90%;
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
        }

        #imageModalClose {
            position: absolute;
            top: 20px;
            right: 35px;
            color: #fff;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
        }
    </style>
</head>

<body>

    <div class="container">
        <h2>AI Generation Studio</h2>

        <div class="input-group">
            <label>Upload Starting Image:</label>
            <div class="upload-area" id="uploadArea">
                <input type="file" id="imageInput" accept="image/png, image/jpeg">
                <div class="upload-text" id="uploadText">
                    Click to upload or drag and drop
                    <small>PNG, JPG up to 10MB</small>
                </div>
                <img id="previewImage" alt="Preview">
            </div>
        </div>

        <div class="input-group">
            <label>Prompt:</label>
            <textarea id="promptInput"
                rows="3">A beautiful cinematic fantasy landscape, vibrant colors, epic volumetric lighting, detailed, ultra high quality, 4k</textarea>
        </div>

        <div class="slider-container">
            <div class="slider-header">
                <label>Creativity:</label>
                <span class="slider-value" id="creativityValue">350</span>
            </div>
            <input type="range" id="creativityInput" min="100" max="600" value="350">
        </div>

        <button id="generateBtn">Generate Magic</button>
        <div id="loading">Processing in Edgeless Cluster...<small>This might take a few seconds</small></div>

        <img id="resultImage" alt="Generated Output">
    </div>

    <!-- Fullscreen image modal -->
    <div id="imageModal">
        <span id="imageModalClose">&times;</span>
        <img id="imageModalContent" alt="Fullscreen">
    </div>

    <div class="container" id="historyPanel">
        <h2>Generation History</h2>
        <button id="historyBtn">Load History</button>
        <div id="historyLoading">Fetching from Edgeless Cluster...<small>This might take a few seconds</small></div>
        <div id="historyContent">
            <div class="history-empty">No history yet. Generate some images first!</div>
        </div>
    </div>

    <script>
        const imageInput = document.getElementById('imageInput');
        const uploadArea = document.getElementById('uploadArea');
        const uploadText = document.getElementById('uploadText');
        const previewImage = document.getElementById('previewImage');
        const creativityInput = document.getElementById('creativityInput');
        const creativityValue = document.getElementById('creativityValue');
        const generateBtn = document.getElementById('generateBtn');
        const loading = document.getElementById('loading');
        const resultImage = document.getElementById('resultImage');

        // Image modal handling
        const modal = document.getElementById('imageModal');
        const modalImg = document.getElementById('imageModalContent');
        const modalClose = document.getElementById('imageModalClose');

        function openImageModal(src) {
            modal.style.display = 'block';
            modalImg.src = src;
        }

        modalClose.addEventListener('click', () => {
            modal.style.display = 'none';
        });

        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.style.display = 'none';
            }
        });

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && modal.style.display === 'block') {
                modal.style.display = 'none';
            }
        });

        // Update slider value display
        creativityInput.addEventListener('input', () => {
            creativityValue.textContent = creativityInput.value;
        });

        // Handle file input click
        uploadArea.addEventListener('click', () => imageInput.click());

        // Handle file selection
        imageInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    previewImage.src = e.target.result;
                    previewImage.style.display = 'block';
                    uploadText.style.display = 'none';
                };
                reader.readAsDataURL(file);
            }
        });

        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                imageInput.files = e.dataTransfer.files;
                const event = new Event('change');
                imageInput.dispatchEvent(event);
            }
        });

        generateBtn.addEventListener('click', async () => {
            const file = imageInput.files[0];
            if (!file) {
                alert("Please upload an image first!");
                return;
            }

            // Lock the UI
            generateBtn.disabled = true;
            loading.style.display = 'block';
            resultImage.style.display = 'none';

            try {
                const reader = new FileReader();
                reader.readAsDataURL(file);

                reader.onload = async () => {
                    const base64String = reader.result.split(',')[1];

                    const payload = {
                        prompt: document.getElementById('promptInput').value,
                        creativity: parseInt(creativityInput.value),
                        image_base64: base64String
                    };

                    const EDGELESS_URL = "/generate";

                    const response = await fetch(EDGELESS_URL, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });

                    if (!response.ok) throw new Error("Server returned " + response.status);

                    const data = await response.json();

                    resultImage.src = "data:image/png;base64," + data.image_base64;
                    resultImage.style.display = 'block';

                    console.log('Generation complete, session_id:', data.session_id);

                    // Prepend generated item to history
                    const now = new Date();
                    const day = String(now.getDate()).padStart(2, '0');
                    const month = String(now.getMonth() + 1).padStart(2, '0');
                    const year = now.getFullYear();
                    const hours = String(now.getHours()).padStart(2, '0');
                    const minutes = String(now.getMinutes()).padStart(2, '0');
                    const seconds = String(now.getSeconds()).padStart(2, '0');
                    const formattedDate = `${day}/${month}/${year}, ${hours}:${minutes}:${seconds}`;

                    const newItem = {
                        id: data.session_id,
                        prompt: payload.prompt,
                        creativity: payload.creativity,
                        source_image_b64: payload.image_base64,
                        generated_image_b64: data.image_base64,
                        created_at: formattedDate
                    };
                    console.log('Calling prependHistoryItem with newItem');
                    prependHistoryItem(newItem);
                    console.log('prependHistoryItem returned, displayedHistoryIds size:', displayedHistoryIds.size);

                    generateBtn.disabled = false;
                    loading.style.display = 'none';
                };
            } catch (error) {
                alert("Error during generation: " + error.message);
                generateBtn.disabled = false;
                loading.style.display = 'none';
            }
        });

        // History button handler
        const historyBtn = document.getElementById('historyBtn');
        const historyContent = document.getElementById('historyContent');
        const historyLoading = document.getElementById('historyLoading');
        let displayedHistoryIds = new Set();

        historyBtn.addEventListener('click', async () => {
            historyBtn.disabled = true;
            historyBtn.classList.add('loading');

            // Always show loading state
            historyLoading.style.display = 'block';
            historyContent.style.display = 'none';

            console.log('Load History clicked, wasEmpty:', wasEmpty);

            try {
                const response = await fetch('/history', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({})
                });

                console.log('Fetch response status:', response.status);

                if (!response.ok) throw new Error("Server returned " + response.status);

                const data = await response.json();
                console.log('Fetch returned data, length:', Array.isArray(data) ? data.length : 'not array');

                historyLoading.style.display = 'none';
                historyContent.style.display = 'block';
                displayHistory(data);
            } catch (error) {
                console.log('Fetch error:', error.message);
                historyLoading.style.display = 'none';
                if (displayedHistoryIds.size === 0) {
                    historyContent.innerHTML = '<div class="history-empty">Error loading history: ' + error.message + '</div>';
                    historyContent.style.display = 'block';
                }
            }

            historyBtn.disabled = false;
            historyBtn.classList.remove('loading');
        });

        function buildHistoryItemHtml(item) {
            const prompt = item.prompt || 'No prompt';
            // Format date consistently: DD/MM/YYYY, HH:MM:SS
            let time = 'Unknown time';
            if (item.created_at) {
                // Parse date in format "YYYY-MM-DD HH:MM:SS" or "DD/MM/YYYY, HH:MM:SS"
                const parts = item.created_at.split(/[\\s,]+/);
                if (parts.length >= 2) {
                    let year, month, day, hours, minutes, seconds;
                    if (parts[0].includes('-')) {
                        // Format: YYYY-MM-DD HH:MM:SS
                        [year, month, day] = parts[0].split('-');
                        [hours, minutes, seconds] = parts[1].split(':');
                    } else {
                        // Format: DD/MM/YYYY HH:MM:SS
                        [day, month, year] = parts[0].split('/');
                        [hours, minutes, seconds] = parts[1].split(':');
                    }
                    if (year && month && day && hours && minutes && seconds) {
                        time = `${day}/${month}/${year}, ${hours}:${minutes}:${seconds}`;
                    }
                }
            }
            const creativity = item.creativity || 'N/A';
            const sourceImg = item.source_image_b64 ? 'data:image/png;base64,' + item.source_image_b64 : '';
            const generatedImg = item.generated_image_b64 ? 'data:image/png;base64,' + item.generated_image_b64 : '';

            return `
                <div class="history-item">
                    <div class="history-item-header">
                        <p class="history-item-prompt" title="Click to expand" onclick="this.classList.toggle('expanded')">${prompt}</p>
                        <span class="history-item-time">${time}</span>
                    </div>
                    <div class="history-item-meta">Creativity: ${creativity}</div>
                    <div class="history-item-images">
                        ${sourceImg ? `<img src="${sourceImg}" alt="Source" title="Click to view" onclick="openImageModal('${sourceImg}')">` : '<img alt="No source">'}
                        ${generatedImg ? `<img src="${generatedImg}" alt="Generated" title="Click to view" onclick="openImageModal('${generatedImg}')">` : '<img alt="No result">'}
                    </div>
                </div>
            `;
        }

        function prependHistoryItem(item) {
            console.log('prependHistoryItem called, item:', item);
            // Clear loading or empty state
            if (historyContent.classList.contains('history-loading')) {
                historyContent.className = '';
                historyContent.innerHTML = '';
            }
            const emptyMsg = historyContent.querySelector('.history-empty');
            if (emptyMsg) {
                emptyMsg.remove();
            }

            // Make sure history panel is visible when adding items
            historyContent.style.display = 'block';

            // Use session_id for deduplication when available (DB items have session_id)
            // For newly generated items, session_id is in item.id
            const itemId = item.session_id || item.id;
            console.log('Adding itemId to displayedHistoryIds:', itemId);
            displayedHistoryIds.add(itemId);
            const newHtml = buildHistoryItemHtml(item);
            historyContent.insertAdjacentHTML('afterbegin', newHtml);
            console.log('History content children count:', historyContent.childNodes.length);
        }

        function displayHistory(items) {
            console.log('displayHistory called, items:', items);
            if (!items || items.length === 0) {
                if (displayedHistoryIds.size === 0) {
                    historyContent.innerHTML = '<div class="history-empty">No history yet. Generate some images first!</div>';
                    historyContent.style.display = 'block';
                }
                return;
            }

            // Filter out already displayed items using session_id (DB items) or id (newly generated)
            // session_id is the UUID that ties DB items to their generated counterparts
            const newItems = items.filter(item => {
                const itemId = item.session_id || item.id;
                return !displayedHistoryIds.has(itemId);
            });
            console.log('newItems after filter:', newItems.length);
            if (newItems.length === 0) {
                return; // Nothing new to display
            }

            // Prepend new items (oldest first so newest appears at top)
            for (const item of newItems.reverse()) {
                prependHistoryItem(item);
            }
        }
    </script>
</body>

</html>
"""


class BridgeHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress HTTP logging

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_UI.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(length))

        # 1. Browser asks for generation
        if self.path == '/generate':
            req_id = str(uuid.uuid4())
            PENDING_TASKS[req_id] = None

            # Forward into Edgeless http-ingress (5 min timeout for AI inference)
            edgeless_payload = {"id": req_id, **body}
            req = urllib.request.Request(
                "http://127.0.0.1:7007/",
                data=json.dumps(edgeless_payload).encode(),
                headers={'Content-Type': 'application/json'}
            )
            urllib.request.urlopen(req, timeout=300)

            # Wait for Edgeless to hit the webhook
            while PENDING_TASKS[req_id] is None:
                time.sleep(0.1)

            result_b64 = PENDING_TASKS.pop(req_id)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"image_base64": result_b64, "session_id": req_id}).encode())

        # 2. Edgeless visualizer sends the result back
        elif self.path == '/webhook':
            req_id = body.get("id")
            if req_id in PENDING_TASKS:
                PENDING_TASKS[req_id] = body.get("image_base64")
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")

        # 3. Browser asks for history
        elif self.path == '/history':
            # Forward history request to Edgeless http-ingress using GET
            # The http-ingress will wrap this into EdgelessHTTPRequest and call sdxl_web_receiver
            req = urllib.request.Request(
                "http://127.0.0.1:7007/history",
                method='GET'
            )
            try:
                response = urllib.request.urlopen(req, timeout=30)
                response_data = response.read().decode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(response_data.encode())
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()


if __name__ == "__main__":
    print(f"Starting web UI server on http://localhost:{PORT}")
    ThreadingHTTPServer(("", PORT), BridgeHandler).serve_forever()