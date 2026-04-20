"""
Flask Web Application — Bone Fracture Detection System (PyTorch)
Accepts an uploaded X-ray image, runs inference + Grad-CAM, returns results.
"""

import os, uuid, base64, io, json
import numpy as np
from flask import Flask, request, render_template, jsonify
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
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB max upload
os.makedirs("uploads", exist_ok=True)
os.makedirs("static",  exist_ok=True)

MODEL_PATH         = "model/fracture_model.pth"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff", "webp"}

# Global model + device — loaded once at startup
model  = None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
def build_inference_model():
    """Rebuild the same architecture used in train.py."""
    m = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    in_features = m.fc.in_features
    # Must match the head defined in train.py exactly
    m.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 1)
        # No Sigmoid — BCEWithLogitsLoss was used in training
        # We apply sigmoid manually during inference
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
        model.eval()   # inference mode: disables dropout, fixes BN stats
        print("   Model loaded successfully.")
    else:
        print(f"⚠️  No model found at {MODEL_PATH}.")
        print("   Run train.py first, or the app will fall back to demo mode.")

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def numpy_to_base64(img_rgb_array):
    """Convert a numpy RGB array to a base64 PNG string."""
    pil_img = Image.fromarray(img_rgb_array.astype(np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def mock_predict(img_path):
    """Fallback demo prediction when no trained model is available."""
    img = np.array(Image.open(img_path).convert("RGB").resize((224, 224)))
    gray    = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    heatmap = cv2.GaussianBlur(gray, (51, 51), 0)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlaid = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
    return "Fractured", 0.87, overlaid

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Please upload PNG, JPG, etc."}), 400

    filename  = f"{uuid.uuid4().hex}.png"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    try:
        if model is not None:
            label, confidence, overlaid = generate_gradcam_image(
                img_path=save_path, model=model, device=device
            )
        else:
            label, confidence, overlaid = mock_predict(save_path)

        result = {
            "label":            label,
            "confidence":       f"{confidence:.1%}",
            "confidence_raw":   round(float(confidence), 4),
            "original_img":     image_to_base64(save_path),
            "gradcam_img":      numpy_to_base64(overlaid),
            "model_used":       "ResNet50 Transfer Learning (PyTorch)" if model else "Demo Mode",
            "recommendation":   get_recommendation(label, confidence)
        }
        return jsonify(result)

    except Exception as e:
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
        "device":       str(device)
    })


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
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    load_model()
    print(f"\n🚀 Starting Fracture Detection Web App  (device: {device})")
    print("   Open http://localhost:5000 in your browser\n")
    app.run(debug=True, host="0.0.0.0", port=5000)