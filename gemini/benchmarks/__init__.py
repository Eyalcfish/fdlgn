from typing import Dict, Callable, Tuple
import torch

from .shifted_substring import get_shifted_substring_dataset
from .tomita4 import get_tomita4_dataset
from .parity32 import get_parity32_dataset
from .majority64 import get_majority64_dataset

BENCHMARK_REGISTRY: Dict[str, Callable] = {
    "shifted_substring": get_shifted_substring_dataset,
    "tomita4": get_tomita4_dataset,
    "parity32": get_parity32_dataset,
    "majority64": get_majority64_dataset,
}

def load_benchmark(name: str, num_train: int = 4000, num_test: int = 1000, seed: int = 42) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], int, int]:
    name = name.lower()
    if name not in BENCHMARK_REGISTRY:
        raise ValueError(f"Unknown benchmark '{name}'. Available: {list(BENCHMARK_REGISTRY.keys())}")
    return BENCHMARK_REGISTRY[name](num_train=num_train, num_test=num_test, seed=seed)
