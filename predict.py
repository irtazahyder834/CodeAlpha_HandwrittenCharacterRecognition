"""
predict.py
----------
Load the trained MNIST CNN model (PyTorch) and predict the digit contained
in a given image file, or provide reusable functions for the Streamlit app.

CLI usage:
    python predict.py path/to/digit_image.png
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model import HandwrittenCNN, get_device

MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "mnist_cnn.pt")
META_PATH = os.path.join(MODELS_DIR, "meta.json")

_model = None
_meta = None
_device = get_device()


def load_model():
    """Load and cache the trained PyTorch CNN model plus its metadata."""
    global _model, _meta
    if _model is None:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run train.py first."
            )
        _model = HandwrittenCNN()
        _model.load_state_dict(torch.load(MODEL_PATH, map_location=_device))
        _model.to(_device)
        _model.eval()
        with open(META_PATH) as f:
            _meta = json.load(f)
    return _model, _meta


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Convert a PIL image into a normalized (1, 1, 28, 28) array matching the
    input the CNN was trained on.
    """
    _, meta = load_model()
    img_dim = meta["img_dim"]

    image = image.convert("L")
    image = image.resize((img_dim, img_dim))
    arr = np.array(image).astype("float32")

    # If background looks light (mean pixel value high), invert so the
    # digit is white on black, matching the training data format.
    if arr.mean() > 255 / 2:
        arr = 255.0 - arr

    arr = arr / 255.0
    return arr.reshape(1, 1, img_dim, img_dim)


def predict_digit(image: Image.Image):
    """
    Predict the digit shown in `image`.

    Returns:
        predicted_label (int), confidence (float), probabilities (np.ndarray)
    """
    model, _ = load_model()
    processed = preprocess_image(image)
    with torch.no_grad():
        tensor = torch.from_numpy(processed).to(_device)
        logits = model(tensor)
        probabilities = F.softmax(logits, dim=1)[0].cpu().numpy()
    predicted_label = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_label])
    return predicted_label, confidence, probabilities


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: file not found: {image_path}")
        sys.exit(1)

    image = Image.open(image_path)
    label, confidence, probs = predict_digit(image)

    print(f"Predicted Digit: {label}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print("Class probabilities:")
    for i, p in enumerate(probs):
        print(f"  {i}: {p * 100:.2f}%")


if __name__ == "__main__":
    main()
