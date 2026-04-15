import torch
import torch.nn as nn
from torchvision import models


class ViolenceClassifier(nn.Module):
    """
    EfficientNet-B0 backbone with temporal mean pooling.
    Input: (B, T, C, H, W) — batch of video clips
    Output: (B, 2) — logits for [NonViolence, Violence]
    """

    def __init__(self, num_frames: int = 16, num_classes: int = 2, dropout: float = 0.4):
        super().__init__()
        self.num_frames = num_frames

        # Feature extractor without the final classifier
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = backbone.features # (B*T, 1280, H', W')
        self.pool = backbone.avgpool  # AdaptiveAvgPool2d (B*T, 1280, 1, 1)
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout), 
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.features(x)
        x = self.pool(x)                   # (B*T, 1280, 1, 1)
        x = x.view(B * T, -1)             # (B*T, 1280)
        x = x.view(B, T, -1).mean(dim=1)  # temporal mean (B, 1280)
        return self.classifier(x)