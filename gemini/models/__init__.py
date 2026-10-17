from typing import Dict, Type
import torch.nn as nn

from .base import count_parameters, GroupSum
from .mlp import MLPModel
from .cnn import CNN1DModel
from .transformer import TransformerModel
from .dlgn import BasicDLGNModel
from .ldlgn import LDLGNModel
from .conv_denseskip_lut4 import ConvSoftDenseSkipLUT4Model

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "mlp": MLPModel,
    "cnn": CNN1DModel,
    "transformer": TransformerModel,
    "dlgn": BasicDLGNModel,
    "ldlgn": LDLGNModel,
    "conv_soft_denseskip_lut4": ConvSoftDenseSkipLUT4Model,
}

def create_model(name: str, in_features: int = 64, num_classes: int = 2, **kwargs) -> nn.Module:
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available models: {list(MODEL_REGISTRY.keys())}")
    model_cls = MODEL_REGISTRY[name]
    return model_cls(in_features=in_features, num_classes=num_classes, **kwargs)
