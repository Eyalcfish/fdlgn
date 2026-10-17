import torch
import torch.nn as nn
from typing import Dict, Any

class GroupSum(nn.Module):
    """
    Readout head for logic gate networks (Petersen et al., 2022).
    Partitions the final layer activations into `num_classes` equal groups and sums each group to produce logits.
    """
    def __init__(self, in_features: int, num_classes: int = 2):
        super().__init__()
        assert in_features % num_classes == 0, f"in_features ({in_features}) must be divisible by num_classes ({num_classes})"
        self.in_features = in_features
        self.num_classes = num_classes
        self.group_size = in_features // num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        # Reshape to [batch, num_classes, group_size] and sum over group_size
        grouped = x.view(batch_size, self.num_classes, self.group_size)
        return grouped.sum(dim=-1) / (self.group_size ** 0.5)

def count_parameters(model: nn.Module) -> int:
    """Returns the total number of trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
