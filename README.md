# Handwritten Character Recognition

A deep learning project that recognizes handwritten digits (0-9) using a
**Convolutional Neural Network (CNN)** built with **PyTorch** and
trained on the MNIST dataset, with an interactive Streamlit web interface
for live predictions.

Built as part of the **CodeAlpha Machine Learning Internship**.

## Project Overview

This project trains a CNN to classify handwritten digit images with high
accuracy, then exposes the trained model through a Streamlit application
where users can upload an image or draw a digit and get an instant
prediction with a confidence score and full probability breakdown.

## Model Architecture

```
Input (28x28x1)
 -> Conv2D(32, 3x3, relu) -> Conv2D(32, 3x3, relu) -> MaxPool(2x2) -> Dropout(0.25)
 -> Conv2D(64, 3x3, relu) -> Conv2D(64, 3x3, relu) -> MaxPool(2x2) -> Dropout(0.25)
 -> Flatten -> Dense(128, relu) -> Dropout(0.5)
 -> Dense(10, softmax)
```

- Framework: PyTorch
- Optimizer: Adam
- Loss: sparse categorical cross-entropy
- Regularization: Dropout + early stopping on a validation split

## Dataset

- **Primary**: the full real **MNIST** dataset — 70,000 grayscale 28x28
  images of handwritten digits 0-9 — downloaded automatically via
  `torchvision.datasets.MNIST(...)` (cached locally after the first run).
- **Automatic offline fallback**: if there is no internet connection,
  `train.py` falls back to `sklearn.datasets.load_digits`, a smaller but
  still real, built-in dataset of 1,797 8x8 handwritten digit images,
  upscaled to 28x28 so the exact same CNN can still be trained end-to-end.
- Either way, the model is trained and evaluated on **real** handwritten
  digit data — never synthetic placeholders.
- `screenshots/classification_report.txt` records which dataset source was
  actually used for the included training run.

## Project Structure

```
Handwritten_Recognition/
│
├── models/                              # Saved trained model (.pt) + metadata
├── screenshots/                         # Auto-generated plots & reports
├── app.py                               # Streamlit GUI
├── train.py                             # Training pipeline
├── predict.py                           # Inference utilities / CLI
├── requirements.txt
├── README.md
└── handwritten_character_recognition.ipynb
```

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/CodeAlpha_HandwrittenRecognition.git
cd CodeAlpha_HandwrittenRecognition

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (saves to models/mnist_cnn.pt)
python train.py

# 4. Launch the Streamlit app
streamlit run app.py
```

## Results

After running `train.py`, the following are generated in `screenshots/`:

- `sample_images.png` — sample training images
- `training_history.png` — training/validation loss & accuracy curves
- `confusion_matrix.png` — confusion matrix heatmap
- `prediction_samples.png` — sample test predictions (correct vs incorrect)
- `classification_report.txt` — precision, recall, F1-score, accuracy

With the full MNIST dataset, this CNN architecture typically reaches
**~99% test accuracy**. The included training run (offline fallback
dataset) reached **98.3% test accuracy** — see `screenshots/` for the
actual generated results.

## Screenshots

See the `screenshots/` folder for generated training curves, confusion
matrix, and prediction sample visualizations from an actual training run
included in this package.

## Future Improvements

- Add EMNIST support for full alphanumeric character recognition.
- Experiment with data augmentation (rotation, shift, zoom) for robustness.
- Extend to a CRNN (CNN + RNN/LSTM) for full word/sentence recognition.
- Deploy as a public Streamlit Cloud / Hugging Face Space demo.

### Repository Name
`CodeAlpha_HandwrittenRecognition`
