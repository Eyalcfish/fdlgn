import torch
import torch.nn as nn
from .base import GroupSum
from .dlgn import sample_wiring

class LDLGNLayer(nn.Module):
    """
    Light DLGN (LDLGN / TT-LUT2, Rüttgers et al., ICLR 2026).
    Direct Input-Wise Parameterization (IWP) of truth tables using only 4 parameters per 2-input gate.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Static random wiring [out_features, 2]
        wires = sample_wiring(in_features, out_features, k=2)
        self.register_buffer("wires", wires)

        # 4 minterm log-odds weights per gate
        self.weights = nn.Parameter(torch.randn(out_features, 4) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, in_features] in [0, 1]
        a = x[:, self.wires[:, 0]]
        b = x[:, self.wires[:, 1]]

        # 4 minterms
        m00 = (1.0 - a) * (1.0 - b)
        m01 = (1.0 - a) * b
        m10 = a * (1.0 - b)
        m11 = a * b
        M = torch.stack([m00, m01, m10, m11], dim=-1) # [batch_size, out_features, 4]

        # Continuous multilinear relaxation: sigmoid maps logits to [0, 1]
        tt = torch.sigmoid(self.weights) # [out_features, 4]

        # Gate output
        out = (M * tt.unsqueeze(0)).sum(dim=-1) # [batch_size, out_features]
        return out

class LDLGNModel(nn.Module):
    """
    Multi-layer Light Differentiable Logic Gate Network (LDLGN) with GroupSum readout.
    """
    def __init__(self, in_features: int = 64, hidden_dim: int = 128, depth: int = 4, num_classes: int = 2):
        super().__init__()
        layers = []
        curr_dim = in_features
        for _ in range(depth):
            layers.append(LDLGNLayer(curr_dim, hidden_dim))
            curr_dim = hidden_dim
        self.layers = nn.Sequential(*layers)
        self.head = GroupSum(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, 0.0, 1.0)
        feats = self.layers(x)
        return self.head(feats)
