"""
Diagnosis script — run this BEFORE fixing anything.
It will tell you exactly what's wrong with inference.

Usage:
    python diagnose.py <path_to_an_xray_you_know_is_fractured>

Example:
    python diagnose.py Dataset/test/fractured/img001.jpg
"""

import sys, json
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image

MODEL_PATH = "model/fracture_model.pth"
CLASS_JSON = "model/class_indices.json"

# ── Rebuild model (must match train.py) ───────────────────────────────────
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

m = models.resnet50(weights=None)
in_features = m.fc.in_features
m.fc = nn.Sequential(
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 1)
)
checkpoint = torch.load(MODEL_PATH, map_location=device)
m.load_state_dict(checkpoint["model_state"])
m.to(device).eval()

print("=" * 55)
print("STEP 1 — Class index map (from training)")
print("=" * 55)
with open(CLASS_JSON) as f:
    class_to_idx = json.load(f)
print(f"  class_to_idx : {class_to_idx}")
print()
print("  ⚠️  ImageFolder sorts alphabetically.")
print("  If 'fractured' → 0 and model outputs logit > 0")
print("  → sigmoid > 0.5 → predicted class index = 1 = NOT fractured!")
print("  That would explain always saying No Fracture.")
print()

# ── Run inference on the provided image ───────────────────────────────────
if len(sys.argv) < 2:
    print("Pass an image path as argument to continue diagnosis.")
    print("Example: python diagnose.py Dataset/test/fractured/some_image.jpg")
    sys.exit(0)

img_path = sys.argv[1]

tf_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

pil_img = Image.open(img_path).convert("RGB")
tensor  = tf_val(pil_img).unsqueeze(0).to(device)

with torch.no_grad():
    logit = m(tensor).item()
    prob  = torch.sigmoid(torch.tensor(logit)).item()

print("=" * 55)
print("STEP 2 — Raw model output on your image")
print("=" * 55)
print(f"  Image path   : {img_path}")
print(f"  Raw logit    : {logit:.4f}")
print(f"  Sigmoid prob : {prob:.4f}")
print()

# ── Interpret ─────────────────────────────────────────────────────────────
print("=" * 55)
print("STEP 3 — Diagnosis")
print("=" * 55)

idx_when_high = 1   # sigmoid > 0.5 → predicted index = 1
idx_when_low  = 0   # sigmoid < 0.5 → predicted index = 0

# find what label corresponds to each index
idx_to_class = {v: k for k, v in class_to_idx.items()}
label_when_high = idx_to_class.get(idx_when_high, "unknown")
label_when_low  = idx_to_class.get(idx_when_low,  "unknown")

print(f"  When prob > 0.5  → model says: '{label_when_high}'")
print(f"  When prob < 0.5  → model says: '{label_when_low}'")
print()

if prob > 0.5:
    print(f"  Current prob = {prob:.4f} → model is saying: '{label_when_high}'")
else:
    print(f"  Current prob = {prob:.4f} → model is saying: '{label_when_low}'")

print()
frac_idx = class_to_idx.get("fractured", class_to_idx.get("Fractured", None))
if frac_idx == 0:
    print("  ✅ FOUND THE BUG:")
    print("     'fractured' maps to index 0, but your app treats")
    print("     prob > 0.5 as fractured. It should be prob < 0.5!")
    print()
    print("  FIX: In gradcam.py, change the threshold logic to:")
    print("       label = 'Fractured' if prob < 0.5 else 'Normal'")
    print("       confidence = 1 - prob if prob < 0.5 else prob")
elif frac_idx == 1:
    print("  ✅ Class mapping looks correct (fractured = index 1).")
    if prob < 0.3:
        print(f"  But prob = {prob:.4f} is very low for a fractured image.")
        print("  This suggests the model may not have trained well enough,")
        print("  or there is a preprocessing mismatch.")
else:
    print("  ⚠️  Could not find 'fractured' key in class_indices.json")
    print(f"     Keys found: {list(class_to_idx.keys())}")
    print("     Check your Dataset folder names match exactly.")

print()
print("=" * 55)
print("STEP 4 — Check your Dataset folder names")
print("=" * 55)
import os
for split in ["train", "val", "test"]:
    split_path = os.path.join("Dataset", split)
    if os.path.exists(split_path):
        folders = sorted(os.listdir(split_path))
        print(f"  Dataset/{split}/ → {folders}")