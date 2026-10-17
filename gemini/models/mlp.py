import torch
import torch.nn as nn

class MLPModel(nn.Module):
    """
    Standard Feedforward Multi-Layer Perceptron baseline with LayerNorm and GELU activations.
    """
    def __init__(self, in_features: int = 64, hidden_dim: int = 128, depth: int = 3, num_classes: int = 2):
        super().__init__()
        layers = []
        curr_dim = in_features
        for _ in range(depth - 1):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, in_features]
        return self.net(x)
