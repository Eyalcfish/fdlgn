import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import GroupSum

# Canonical 4x16 truth-table matrix for all 16 2-input Boolean operators
# Rows: minterms m00, m01, m10, m11
# Columns: operator index 0 to 15
TRUTH_TABLE_16 = torch.tensor([
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], # m00: a=0, b=0
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1], # m01: a=0, b=1
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1], # m10: a=1, b=0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1], # m11: a=1, b=1
], dtype=torch.float32)

def sample_wiring(in_features: int, out_features: int, k: int = 2) -> torch.Tensor:
    wires = torch.zeros(out_features, k, dtype=torch.long)
    for i in range(out_features):
        if in_features >= k:
            wires[i] = torch.randperm(in_features)[:k]
        else:
            wires[i] = torch.randint(0, in_features, (k,))
    return wires

class DLGNLayer(nn.Module):
    """
    Standard Differentiable Logic Gate Network layer (Petersen et al., NeurIPS 2022).
    Uses a 16-way categorical softmax distribution over the 16 2-input Boolean operators.
    """
    def __init__(self, in_features: int, out_features: int, tau: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tau = tau

        # Static random wiring [out_features, 2]
        wires = sample_wiring(in_features, out_features, k=2)
        self.register_buffer("wires", wires)
        self.register_buffer("TT", TRUTH_TABLE_16)

        # 16-way categorical logits per gate
        self.weights = nn.Parameter(torch.randn(out_features, 16) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, in_features] in [0, 1]
        a = x[:, self.wires[:, 0]] # [batch_size, out_features]
        b = x[:, self.wires[:, 1]] # [batch_size, out_features]

        # 4 minterm probabilities
        m00 = (1.0 - a) * (1.0 - b)
        m01 = (1.0 - a) * b
        m10 = a * (1.0 - b)
        m11 = a * b
        M = torch.stack([m00, m01, m10, m11], dim=-1) # [batch_size, out_features, 4]

        # Categorical operator probabilities [out_features, 16]
        alpha = F.softmax(self.weights / self.tau, dim=-1)

        # Effective truth table for each gate: [out_features, 4] = alpha @ TT.T
        eff_tt = torch.matmul(alpha, self.TT.t()) # [out_features, 4]

        # Gate outputs: sum over the 4 minterms
        out = (M * eff_tt.unsqueeze(0)).sum(dim=-1) # [batch_size, out_features]
        return out

class BasicDLGNModel(nn.Module):
    """
    Multi-layer Differentiable Logic Gate Network (Petersen et al., 2022) with GroupSum readout.
    """
    def __init__(self, in_features: int = 64, hidden_dim: int = 128, depth: int = 4, num_classes: int = 2):
        super().__init__()
        layers = []
        curr_dim = in_features
        for _ in range(depth):
            layers.append(DLGNLayer(curr_dim, hidden_dim))
            curr_dim = hidden_dim
        self.layers = nn.Sequential(*layers)
        self.head = GroupSum(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, 0.0, 1.0)
        feats = self.layers(x)
        return self.head(feats)
