"""
model.py
--------
PyTorch CNN architecture for the Handwritten Character Recognition project.
Shared by train.py, predict.py, and app.py so the exact same architecture is
used for training and inference.

Architecture:
    Conv2d(32) -> Conv2d(32) -> MaxPool -> Dropout
    Conv2d(64) -> Conv2d(64) -> MaxPool -> Dropout
    Flatten -> Linear(128) -> Dropout -> Linear(10)
"""

import torch
import torch.nn as nn

IMG_DIM = 28
NUM_CLASSES = 10


class HandwrittenCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (IMG_DIM // 4) * (IMG_DIM // 4), 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (batch, 1, IMG_DIM, IMG_DIM) -> raw logits (no softmax; use
        # nn.CrossEntropyLoss for training and torch.softmax for probabilities)
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
