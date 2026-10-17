import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]

class TransformerModel(nn.Module):
    """
    Sequence Transformer baseline evaluating self-attention over binary register sequences.
    """
    def __init__(self, in_features: int = 64, d_model: int = 32, nhead: int = 4, num_layers: int = 2, num_classes: int = 2):
        super().__init__()
        self.in_features = in_features
        # Linear projection of each bit into d_model embedding space
        self.bit_embed = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=in_features)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.0,
            batch_first=True,
            activation="gelu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, in_features]
        # Treat each bit as a token: [batch_size, in_features, 1]
        tokens = x.unsqueeze(-1)
        h = self.bit_embed(tokens) # [batch_size, in_features, d_model]
        h = self.pos_encoder(h)
        h = self.transformer_encoder(h)
        h = self.norm(h)
        # Global mean pooling across tokens
        pooled = h.mean(dim=1)
        return self.classifier(pooled)
