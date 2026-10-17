import numpy as np
import torch
from typing import Tuple, Set

DEFAULT_NEEDLE = (1, 0, 1, 1, 0, 0, 1, 0)

def contains_needle(bits: np.ndarray, needle: np.ndarray) -> bool:
    needle_len = len(needle)
    for i in range(len(bits) - needle_len + 1):
        if np.array_equal(bits[i : i + needle_len], needle):
            return True
    return False

def get_shifted_substring_dataset(
    num_train: int = 4000,
    num_test: int = 1000,
    reg_len: int = 64,
    needle_len: int = 8,
    seed: int = 42
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], int, int]:
    """
    Shifted Substring Benchmark:
    8-bit needle hidden at a random offset within a 64-bit register.
    Balanced 50/50 classes with strict zero data leakage between train and test.
    """
    rng = np.random.RandomState(seed)
    needle = np.array(DEFAULT_NEEDLE, dtype=np.float32)

    total_samples = num_train + num_test
    total_pos = total_samples // 2
    total_neg = total_samples - total_pos

    max_offset = reg_len - needle_len

    seen_all: Set[Tuple[int, ...]] = set()
    pos_samples = []
    while len(pos_samples) < total_pos:
        offset = rng.randint(0, max_offset + 1)
        sample = rng.randint(0, 2, size=reg_len).astype(np.float32)
        sample[offset : offset + needle_len] = needle
        tup = tuple(sample.astype(int))
        if tup not in seen_all:
            seen_all.add(tup)
            pos_samples.append(sample)

    neg_samples = []
    while len(neg_samples) < total_neg:
        sample = rng.randint(0, 2, size=reg_len).astype(np.float32)
        if not contains_needle(sample, needle):
            tup = tuple(sample.astype(int))
            if tup not in seen_all:
                seen_all.add(tup)
                neg_samples.append(sample)

    pos_samples = np.array(pos_samples, dtype=np.float32)
    neg_samples = np.array(neg_samples, dtype=np.float32)

    train_pos = num_train // 2
    train_neg = num_train - train_pos
    test_pos = num_test // 2
    test_neg = num_test - test_pos

    X_train = np.vstack([pos_samples[:train_pos], neg_samples[:train_neg]])
    y_train = np.concatenate([np.ones(train_pos, dtype=np.int64), np.zeros(train_neg, dtype=np.int64)])

    X_test = np.vstack([pos_samples[train_pos : train_pos + test_pos], neg_samples[train_neg : train_neg + test_neg]])
    y_test = np.concatenate([np.ones(test_pos, dtype=np.int64), np.zeros(test_neg, dtype=np.int64)])

    # Shuffle splits
    idx_tr = rng.permutation(len(X_train))
    X_train, y_train = X_train[idx_tr], y_train[idx_tr]

    idx_te = rng.permutation(len(X_test))
    X_test, y_test = X_test[idx_te], y_test[idx_te]

    return (
        (torch.from_numpy(X_train), torch.from_numpy(y_train)),
        (torch.from_numpy(X_test), torch.from_numpy(y_test)),
        reg_len,
        2
    )
