"""
Grad-CAM: Gradient-weighted Class Activation Mapping (PyTorch)
Highlights which region of the X-ray triggered the fracture prediction.
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


# ─────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────
# Must match the val_transforms used in train.py exactly
INFERENCE_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    ),
])


# ─────────────────────────────────────────────
# GRAD-CAM CORE
# ─────────────────────────────────────────────
class GradCAM:
    """
    Hooks into the last conv layer of ResNet50 (layer4) and captures:
      - forward activations (feature maps)
      - backward gradients
    Then weights the feature maps by the gradients to produce a heatmap.
    """

    def __init__(self, model, device):
        self.model   = model
        self.device  = device
        self._activations = None
        self._gradients   = None

        # layer4 is the last residual block group in ResNet50
        # This is equivalent to conv5_block3_out in the TF/Keras version
        target_layer = model.layer4

        # Forward hook — saves feature map output
        self._fwd_hook = target_layer.register_forward_hook(
            lambda m, inp, out: self._save_activation(out)
        )
        # Backward hook — saves gradients flowing back through the layer
        self._bwd_hook = target_layer.register_full_backward_hook(
            lambda m, grad_in, grad_out: self._save_gradient(grad_out[0])
        )

    def _save_activation(self, activation):
        self._activations = activation.detach()

    def _save_gradient(self, gradient):
        self._gradients = gradient.detach()

    def generate(self, input_tensor):
        """
        Run forward + backward pass and compute the Grad-CAM heatmap.

        Args:
            input_tensor: (1, 3, 224, 224) preprocessed image tensor on device

        Returns:
            heatmap: numpy array (H, W) with values in [0, 1]
            confidence: float, probability of fracture
            label: str, "Fractured" or "Normal"
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass — logits output (no sigmoid in model head)
        logits = self.model(input_tensor)            # shape: (1, 1)
        prob   = torch.sigmoid(logits).item()        # apply sigmoid for probability

        # fractured=0, not fractured=1 in training (alphabetical sort)
        # so LOW prob = fractured, HIGH prob = not fractured
        label      = "Fractured" if prob < 0.5 else "Normal"
        confidence = 1 - prob if prob < 0.5 else prob

        # Backward pass on the raw logit (not sigmoid) — standard Grad-CAM practice
        logits.backward()

        # Global average pool the gradients over spatial dims → importance weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activation maps
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)                            # keep only positive influence
        cam = cam.squeeze().cpu().numpy()            # (H, W)

        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam, confidence, label

    def remove_hooks(self):
        """Call this when done to free memory."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ─────────────────────────────────────────────
# OVERLAY
# ─────────────────────────────────────────────
def overlay_gradcam(original_img_rgb, heatmap, alpha=0.4):
    """
    Overlay the Grad-CAM heatmap on the original X-ray image.

    Args:
        original_img_rgb : numpy (H, W, 3) uint8 RGB
        heatmap          : numpy (h, w) float 0–1
        alpha            : heatmap transparency

    Returns:
        overlaid image as numpy (H, W, 3) uint8 RGB
    """
    h, w = original_img_rgb.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlaid = cv2.addWeighted(original_img_rgb, 1 - alpha, heatmap_colored, alpha, 0)
    return overlaid


# ─────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────
def generate_gradcam_image(img_path, model, device, save_path=None):
    """
    Full pipeline: load image → predict → Grad-CAM → overlay → return.

    Args:
        img_path  : path to the input X-ray image
        model     : loaded PyTorch model (eval mode, on device)
        device    : torch.device
        save_path : optional path to save the overlaid image

    Returns:
        (label, confidence, overlaid_image_rgb)
    """
    # Load image — keep original RGB copy for overlay
    pil_img      = Image.open(img_path).convert("RGB")
    img_rgb      = np.array(pil_img.resize((224, 224)))  # (224, 224, 3) uint8

    # Preprocess for model
    input_tensor = INFERENCE_TRANSFORMS(pil_img).unsqueeze(0).to(device)  # (1,3,224,224)

    # Run Grad-CAM
    gradcam = GradCAM(model, device)
    try:
        heatmap, confidence, label = gradcam.generate(input_tensor)
    finally:
        gradcam.remove_hooks()  # always clean up hooks

    overlaid = overlay_gradcam(img_rgb, heatmap)

    if save_path:
        out_bgr = cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, out_bgr)

    return label, confidence, overlaid


# ─────────────────────────────────────────────
# Quick standalone test
# python gradcam.py model/fracture_model.pth path/to/xray.jpg
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    from torchvision import models
    from torchvision.models import ResNet50_Weights
    import torch.nn as nn

    if len(sys.argv) < 3:
        print("Usage: python gradcam.py <model_path> <image_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    img_path   = sys.argv[2]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Rebuild architecture (must match train.py)
    m = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    in_features = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 1)
    )
    checkpoint = torch.load(model_path, map_location=device)
    m.load_state_dict(checkpoint["model_state"])
    m.to(device).eval()

    label, confidence, overlaid = generate_gradcam_image(
        img_path, m, device, save_path="gradcam_result.png"
    )

    print(f"Prediction : {label}")
    print(f"Confidence : {confidence:.2%}")
    print(f"Grad-CAM   : saved to gradcam_result.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(Image.open(img_path))
    axes[0].set_title("Original X-Ray");  axes[0].axis("off")
    axes[1].imshow(overlaid)
    axes[1].set_title(f"Grad-CAM  |  {label}  ({confidence:.1%})")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig("gradcam_comparison.png", dpi=150)
    plt.show()