"""
train.py
--------
Trains a Convolutional Neural Network (CNN) with PyTorch on the MNIST
handwritten digit dataset, evaluates it, saves all diagnostic plots to
`screenshots/`, and saves the trained model to `models/mnist_cnn.pt`.

Architecture:
    Conv2D(32) -> Conv2D(32) -> MaxPool -> Dropout
    Conv2D(64) -> Conv2D(64) -> MaxPool -> Dropout
    Flatten -> Dense(128) -> Dropout -> Dense(10, softmax)

Dataset loading strategy:
  1. Try to load the full real MNIST dataset (70,000 28x28 grayscale
     images) via torchvision.datasets.MNIST. This downloads the official
     dataset on first run (needs internet) and caches it locally
     afterwards (./mnist_data).
  2. If that fails (no internet access, e.g. in a sandboxed environment),
     automatically fall back to scikit-learn's `load_digits` -- a smaller
     but still genuinely real, offline 8x8 handwritten digit dataset --
     and upscale each image to 28x28 so the exact same CNN architecture
     and pipeline can still be trained and evaluated end-to-end.

Usage:
    python train.py
"""

import os
import json
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import torchvision

from model import HandwrittenCNN, IMG_DIM, NUM_CLASSES, get_device

MODELS_DIR = "models"
SCREENSHOTS_DIR = "screenshots"
MODEL_PATH = os.path.join(MODELS_DIR, "mnist_cnn.pt")
META_PATH = os.path.join(MODELS_DIR, "meta.json")
RANDOM_SEED = 42

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = get_device()


def _upscale_to_28(images_8x8: np.ndarray) -> np.ndarray:
    """Upscale an (N, 8, 8) array of 0-16 digit images to (N, 28, 28) 0-255."""
    out = np.zeros((len(images_8x8), IMG_DIM, IMG_DIM), dtype="uint8")
    for i, img in enumerate(images_8x8):
        pil_img = Image.fromarray((img / 16.0 * 255).astype("uint8"))
        pil_img = pil_img.resize((IMG_DIM, IMG_DIM), Image.BICUBIC)
        out[i] = np.array(pil_img)
    return out


def load_dataset():
    """
    Load real handwritten digit data as (X_train, y_train, X_test, y_test),
    with images shaped (N, 1, 28, 28) scaled to [0, 1].

    Returns also a `source` string describing which dataset was used.
    """
    try:
        print("Attempting to download the full real MNIST dataset "
              "(70,000 28x28 images) via torchvision.datasets.MNIST...")
        train_set = torchvision.datasets.MNIST(root="mnist_data", train=True, download=True)
        test_set = torchvision.datasets.MNIST(root="mnist_data", train=False, download=True)
        X_train = train_set.data.numpy()
        y_train = train_set.targets.numpy()
        X_test = test_set.data.numpy()
        y_test = test_set.targets.numpy()
        print(f"Loaded full MNIST: {len(X_train)} train / {len(X_test)} test images.")
        source = "full MNIST (70,000 images, 28x28)"
    except Exception as exc:
        print(f"Could not download full MNIST ({exc}).")
        print("Falling back to scikit-learn's built-in 'load_digits' dataset "
              "(1,797 real handwritten digit images, no internet required), "
              "upscaled from 8x8 to 28x28 so the same CNN can be used.")
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split

        digits = load_digits()
        X_all = _upscale_to_28(digits.images)
        y_all = digits.target.astype("int64")
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2, random_state=RANDOM_SEED, stratify=y_all
        )
        source = "sklearn load_digits (1,797 images, upscaled 8x8 -> 28x28)"

    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    X_train = X_train.reshape(-1, 1, IMG_DIM, IMG_DIM)
    X_test = X_test.reshape(-1, 1, IMG_DIM, IMG_DIM)

    return X_train, y_train.astype("int64"), X_test, y_test.astype("int64"), source


def visualize_samples(X, y, n=10):
    plt.figure(figsize=(12, 3))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(X[i].reshape(IMG_DIM, IMG_DIM), cmap="gray")
        plt.title(str(y[i]))
        plt.axis("off")
    plt.suptitle("Sample Handwritten Digit Training Images")
    plt.tight_layout()
    plt.savefig(os.path.join(SCREENSHOTS_DIR, "sample_images.png"), dpi=150)
    plt.close()


def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history["loss"], label="Training Loss", color="crimson")
    axes[0].plot(history["val_loss"], label="Validation Loss", color="darkorange")
    axes[0].set_title("Model Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history["accuracy"], label="Training Accuracy", color="seagreen")
    axes[1].plot(history["val_accuracy"], label="Validation Accuracy", color="royalblue")
    axes[1].set_title("Model Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(SCREENSHOTS_DIR, "training_history.png"), dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=range(NUM_CLASSES), yticklabels=range(NUM_CLASSES))
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(SCREENSHOTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()


def plot_prediction_samples(X_test, y_true, y_pred, n=10):
    plt.figure(figsize=(15, 3))
    idxs = np.random.choice(len(X_test), n, replace=False)
    for i, idx in enumerate(idxs):
        plt.subplot(1, n, i + 1)
        plt.imshow(X_test[idx].reshape(IMG_DIM, IMG_DIM), cmap="gray")
        color = "green" if y_true[idx] == y_pred[idx] else "red"
        plt.title(f"T:{y_true[idx]} P:{y_pred[idx]}", color=color)
        plt.axis("off")
    plt.suptitle("Prediction Samples (Green = Correct, Red = Incorrect)")
    plt.tight_layout()
    plt.savefig(os.path.join(SCREENSHOTS_DIR, "prediction_samples.png"), dpi=150)
    plt.close()


def train_model(model, train_loader, val_loader, epochs=25, patience=5):
    """Train with Adam + CrossEntropyLoss, manual early stopping that
    restores the best validation-loss weights (mirrors Keras'
    EarlyStopping(monitor='val_loss', patience=..., restore_best_weights=True))."""
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            running_correct += (outputs.argmax(1) == yb).sum().item()
            running_total += xb.size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                outputs = model(xb)
                loss = criterion(outputs, yb)
                val_loss += loss.item() * xb.size(0)
                val_correct += (outputs.argmax(1) == yb).sum().item()
                val_total += xb.size(0)
        val_loss /= val_total
        val_acc = val_correct / val_total

        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        print(f"Epoch {epoch + 1}/{epochs} - loss: {train_loss:.4f} - accuracy: {train_acc:.4f} "
              f"- val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1} (best val_loss: {best_val_loss:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def main():
    print("Loading handwritten digit dataset...")
    X_train, y_train, X_test, y_test, source = load_dataset()
    print(f"Dataset source: {source}")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    print("Saving sample image visualization...")
    visualize_samples(X_train, y_train)

    print("Building CNN model...")
    model = HandwrittenCNN().to(DEVICE)
    print(model)

    full_train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_size = int(0.1 * len(full_train_ds))
    train_size = len(full_train_ds) - val_size
    train_ds, val_ds = random_split(
        full_train_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    print("Training model...")
    history = train_model(model, train_loader, val_loader, epochs=25, patience=5)

    print("Plotting training history...")
    plot_training_history(history)

    print("Evaluating model on test set...")
    model.eval()
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=256, shuffle=False,
    )
    all_preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(DEVICE)
            outputs = model(xb)
            all_preds.append(outputs.argmax(1).cpu().numpy())
    y_pred = np.concatenate(all_preds)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_acc:.4f}")

    report = classification_report(y_test, y_pred, digits=4)
    print("Classification Report:")
    print(report)
    with open(os.path.join(SCREENSHOTS_DIR, "classification_report.txt"), "w") as f:
        f.write(f"Dataset source: {source}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
        f.write(report)

    print("Plotting confusion matrix...")
    plot_confusion_matrix(y_test, y_pred)

    print("Plotting prediction samples...")
    plot_prediction_samples(X_test, y_test, y_pred)

    print(f"Saving trained model to {MODEL_PATH} ...")
    torch.save(model.state_dict(), MODEL_PATH)
    with open(META_PATH, "w") as f:
        json.dump({"img_dim": IMG_DIM, "pixel_max": 255.0, "dataset_source": source}, f)

    print("Done. Model and evaluation artifacts saved.")


if __name__ == "__main__":
    main()
