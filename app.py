"""
Flask Web Application — Bone Fracture Detection System (PyTorch)
Accepts an uploaded X-ray image, runs inference + Grad-CAM, returns results.

FIXES APPLIED:
  1. Serves index.html directly from the same folder as app.py — no templates/ folder needed.
  2. Global error handlers always return JSON, never HTML.
  3. Lazy model loading — model loads on first request, not at startup (fixes Render OOM).
  4. Image re-encoded as PNG on save to handle all formats cleanly.
  5. debug=False to avoid double model load.
"""

import os, uuid, base64, io, traceback
import numpy as np
from flask import Flask, request, jsonify, send_file
from PIL import Image
import cv2

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

from gradcam import generate_gradcam_image

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")
app.config["UPLOAD_FOLDER"]      = os.path.join(BASE_DIR, "uploads")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB max upload

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "static"), exist_ok=True)

MODEL_PATH         = os.path.join(BASE_DIR, "model", "fracture_model.pth")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff", "webp"}

# Global model + device
model  = None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# GLOBAL JSON ERROR HANDLERS
# Prevents Flask from ever returning an HTML error page.
# ─────────────────────────────────────────────

@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": f"Bad request: {str(e)}"}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum upload size is 16 MB."}), 413

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.errorhandler(Exception)
def unhandled_exception(e):
    traceback.print_exc()
    return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────

def build_inference_model():
    m = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    in_features = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 1)
    )
    return m

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        print(f"✅ Loading model from {MODEL_PATH}  (device: {device})...")
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model = build_inference_model()
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        model.eval()
        print("   Model loaded successfully.")
    else:
        print(f"⚠️  No model found at {MODEL_PATH}.")
        print("   Falling back to demo mode.")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def numpy_to_base64(img_rgb_array):
    pil_img = Image.fromarray(img_rgb_array.astype(np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def mock_predict(img_path):
    """Fallback demo prediction when no trained model is available."""
    img      = np.array(Image.open(img_path).convert("RGB").resize((224, 224)))
    gray     = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    heatmap  = cv2.GaussianBlur(gray, (51, 51), 0)
    heatmap  = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    colored  = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    colored  = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlaid = cv2.addWeighted(img, 0.6, colored, 0.4, 0)
    return "Fractured", 0.87, overlaid

def get_recommendation(label, confidence):
    if label == "Fractured":
        if confidence > 0.90:
            return "High confidence fracture detected. Immediate radiologist review recommended."
        elif confidence > 0.75:
            return "Probable fracture detected. Radiologist confirmation advised."
        else:
            return "Possible fracture detected. Further imaging may be required."
    else:
        if confidence > 0.90:
            return "No fracture detected with high confidence. Routine follow-up as needed."
        else:
            return "No fracture detected. Clinical correlation recommended."


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    """
    Serve index.html directly from BASE_DIR (same folder as app.py).
    No templates/ subdirectory required.
    """
    index_path = os.path.join(BASE_DIR, "index.html")
    if not os.path.exists(index_path):
        return jsonify({"error": "index.html not found. Place it in the same folder as app.py."}), 404
    return send_file(index_path)


@app.route("/predict", methods=["POST"])
def predict():
    # ── Lazy load model on first request ─────
    global model
    if model is None:
        load_model()

    # ── Validate upload ───────────────────────
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request. Make sure the form field is named 'file'."}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Please upload PNG, JPG, JPEG, BMP, or TIFF."}), 400

    # ── Save uploaded file ────────────────────
    filename  = f"{uuid.uuid4().hex}.png"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        # Re-encode as PNG regardless of original format (fixes palette/mode issues)
        pil_img = Image.open(file.stream).convert("RGB")
        pil_img.save(save_path, format="PNG")
    except Exception as e:
        return jsonify({"error": f"Could not read image file: {str(e)}"}), 400

    # ── Run inference ─────────────────────────
    try:
        if model is not None:
            label, confidence, overlaid = generate_gradcam_image(
                img_path=save_path, model=model, device=device
            )
        else:
            label, confidence, overlaid = mock_predict(save_path)

        result = {
            "label":          label,
            "confidence":     f"{confidence:.1%}",
            "confidence_raw": round(float(confidence), 4),
            "original_img":   image_to_base64(save_path),
            "gradcam_img":    numpy_to_base64(overlaid),
            "model_used":     "ResNet50 Transfer Learning (PyTorch)" if model else "Demo Mode",
            "recommendation": get_recommendation(label, confidence),
        }
        return jsonify(result), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(save_path):
            os.remove(save_path)


@app.route("/health")
def health():
    return jsonify({
        "status":       "running",
        "model_loaded": model is not None,
        "model_path":   MODEL_PATH,
        "device":       str(device),
    })


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n🚀 Starting Fracture Detection Web App  (device: {device})")
    print("   Open http://localhost:5000 in your browser\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
