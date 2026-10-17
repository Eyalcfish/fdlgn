import torch
import torch.nn as nn

class CNN1DModel(nn.Module):
    """
    1D Convolutional Neural Network baseline for binary sequence / register pattern recognition.
    """
    def __init__(self, in_features: int = 64, num_classes: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(4)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4, 64),
            nn.GELU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, in_features]
        x = x.unsqueeze(1) # [batch_size, 1, in_features]
        feats = self.conv(x)
        flat = feats.view(feats.size(0), -1)
        return self.classifier(flat)
