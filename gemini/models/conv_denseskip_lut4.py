import torch
import torch.nn as nn
from typing import Optional, List
from .base import GroupSum

def sample_connections_k(in_features: int, out_features: int, k: int = 4) -> torch.Tensor:
    wires = torch.zeros(out_features, k, dtype=torch.long)
    for i in range(out_features):
        if in_features >= k:
            wires[i] = torch.randperm(in_features)[:k]
        else:
            wires[i] = torch.randint(0, in_features, (k,))
    return wires

class TruthTableLUT4Layer(nn.Module):
    """
    4-input Look-Up Table layer with Input-Wise Parameterization (IWP / Light DLGN, 16 minterm parameters per gate).
    Computes fast 2x2 factorized bilinear tensor contraction:
      p01 = [(1-u0)(1-u1), (1-u0)u1, u0(1-u1), u0*u1] (shape B x N x 4)
      p23 = [(1-u2)(1-u3), (1-u2)u3, u2(1-u3), u2*u3] (shape B x N x 4)
      outer product basis M = p01.unsqueeze(-1) * p23.unsqueeze(-2) -> flattened to [B, N, 16]
      output = (M * sigmoid(weights)).sum(dim=-1)
    """
    def __init__(self, in_features: int, out_features: int, wires: Optional[torch.Tensor] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        if wires is None:
            wires = sample_connections_k(in_features, out_features, k=4)
        self.register_buffer("wires", wires)

        # 16 minterm log-odds weights per gate
        self.weights = nn.Parameter(torch.randn(out_features, 16) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, in_features]
        # Gather 4 inputs per gate
        u0 = x[:, self.wires[:, 0]]
        u1 = x[:, self.wires[:, 1]]
        u2 = x[:, self.wires[:, 2]]
        u3 = x[:, self.wires[:, 3]]

        # Factorized bilinear minterms
        p01 = torch.stack([
            (1.0 - u0) * (1.0 - u1),
            (1.0 - u0) * u1,
            u0 * (1.0 - u1),
            u0 * u1
        ], dim=-1) # [B, N, 4]

        p23 = torch.stack([
            (1.0 - u2) * (1.0 - u3),
            (1.0 - u2) * u3,
            u2 * (1.0 - u3),
            u2 * u3
        ], dim=-1) # [B, N, 4]

        # Outer product basis [B, N, 16]
        M = (p01.unsqueeze(-1) * p23.unsqueeze(-2)).flatten(start_dim=-2)

        # Continuous soft relaxation
        theta = torch.sigmoid(self.weights) # [N, 16]
        out = (M * theta.unsqueeze(0)).sum(dim=-1)
        return out

class Conv1DLUT4Layer(nn.Module):
    """
    1D Sliding Tree-Convolution layer with spatial Logical OR-Pooling (argus-dlgn-research Exp21/22).
    Evaluates 4-input / 8-input receptive fields across sequence positions, followed by soft OR-pooling over positions.
    """
    def __init__(self, in_seq_len: int = 64, num_filters: int = 32, kernel_size: int = 8, stride: int = 1):
        super().__init__()
        self.in_seq_len = in_seq_len
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride

        self.num_positions = (in_seq_len - kernel_size) // stride + 1
        assert self.num_positions > 0, f"Sequence length {in_seq_len} too small for kernel {kernel_size}"

        # Level 1: 2 LUT4 gates per filter evaluating left [0..3] and right [4..7] halves of window
        self.lut_left = nn.Parameter(torch.randn(num_filters, 16) * 0.1)
        self.lut_right = nn.Parameter(torch.randn(num_filters, 16) * 0.1)
        # Level 2 (Root): combines (left_out, right_out, win[0], win[7])
        self.lut_root = nn.Parameter(torch.randn(num_filters, 16) * 0.1)

    def _eval_lut4(self, u0: torch.Tensor, u1: torch.Tensor, u2: torch.Tensor, u3: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # u: [B, num_pos, num_filters]
        p01 = torch.stack([
            (1.0 - u0) * (1.0 - u1),
            (1.0 - u0) * u1,
            u0 * (1.0 - u1),
            u0 * u1
        ], dim=-1)
        p23 = torch.stack([
            (1.0 - u2) * (1.0 - u3),
            (1.0 - u2) * u3,
            u2 * (1.0 - u3),
            u2 * u3
        ], dim=-1)
        M = (p01.unsqueeze(-1) * p23.unsqueeze(-2)).flatten(start_dim=-2) # [B, pos, filters, 16]
        theta = torch.sigmoid(weights).view(1, 1, self.num_filters, 16)
        return (M * theta).sum(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_seq_len]
        batch_size = x.size(0)

        # Unfold sliding windows: [B, num_positions, kernel_size]
        windows = x.unfold(dimension=1, size=self.kernel_size, step=self.stride) # [B, num_pos, K]
        # Expand across filters: [B, num_pos, num_filters]
        def get_pin(idx):
            return windows[:, :, idx].unsqueeze(-1).expand(-1, -1, self.num_filters)

        # Level 1 evaluation
        left_out = self._eval_lut4(get_pin(0), get_pin(1), get_pin(2), get_pin(3), self.lut_left)
        right_out = self._eval_lut4(get_pin(4), get_pin(5), get_pin(6), get_pin(7), self.lut_right)

        # Level 2 (Root) evaluation
        root_out = self._eval_lut4(left_out, right_out, get_pin(0), get_pin(7), self.lut_root) # [B, num_pos, num_filters]

        # Spatial Logical OR-Pooling: 1.0 - prod_t(1.0 - root_out)
        clamped = torch.clamp(root_out, 0.0, 1.0)
        pooled = 1.0 - torch.prod(1.0 - clamped + 1e-7, dim=1)
        return pooled # [B, num_filters]

class DenseSkipLUT4Backbone(nn.Module):
    """
    DenseSkip-1 LUT-4 Backbone:
    Each layer l samples 3 inputs from layer l-1 and 1 multi-hop skip input from all prior layers {0..l-1} + initial input.
    """
    def __init__(self, in_features: int, hidden_dim: int = 128, depth: int = 4):
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList()

        # Cumulative feature tracking
        prior_features = in_features

        for l in range(depth):
            wires = torch.zeros(hidden_dim, 4, dtype=torch.long)
            for g in range(hidden_dim):
                # 3 inputs from previous layer (or inputs for l=0)
                if l == 0:
                    wires[g, :3] = torch.randperm(in_features)[:3] if in_features >= 3 else torch.randint(0, in_features, (3,))
                    wires[g, 3] = torch.randint(0, in_features, (1,))
                else:
                    # 3 inputs from immediate prior layer l-1
                    prev_start = prior_features - hidden_dim
                    wires[g, :3] = prev_start + torch.randperm(hidden_dim)[:3]
                    # 1 skip input from ANY prior layer or initial input [0 .. prior_features - hidden_dim)
                    wires[g, 3] = torch.randint(0, prev_start, (1,))

            self.layers.append(TruthTableLUT4Layer(prior_features, hidden_dim, wires=wires))
            prior_features += hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cumulative feature buffer: [B, total_wires]
        history = [x]
        curr = x
        for layer in self.layers:
            combined = torch.cat(history, dim=-1)
            curr = layer(combined)
            history.append(curr)
        return curr # Returns activations of the final layer

class ConvSoftDenseSkipLUT4Model(nn.Module):
    """
    Full Conv Soft DenseSkip-1 LUT4 Architecture from argus-dlgn-research (Exp21/22).
    Combines sliding tree-convolutions with OR-pooling, multi-hop DenseSkip backbone, and GroupSum readout.
    """
    def __init__(self, in_features: int = 64, num_filters: int = 32, hidden_dim: int = 128, depth: int = 4, num_classes: int = 2):
        super().__init__()
        self.conv = Conv1DLUT4Layer(in_seq_len=in_features, num_filters=num_filters, kernel_size=8, stride=1)
        # Backbone receives both pooled conv features AND original raw inputs
        backbone_in = num_filters + in_features
        self.backbone = DenseSkipLUT4Backbone(in_features=backbone_in, hidden_dim=hidden_dim, depth=depth)
        self.head = GroupSum(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, 0.0, 1.0)
        conv_feats = self.conv(x)
        combined_in = torch.cat([conv_feats, x], dim=-1)
        deep_feats = self.backbone(combined_in)
        return self.head(deep_feats)
