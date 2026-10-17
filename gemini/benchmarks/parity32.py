import numpy as np
import torch
from typing import Tuple, Set

def get_parity32_dataset(
    num_train: int = 4000,
    num_test: int = 1000,
    active_bits: int = 32,
    reg_len: int = 64,
    seed: int = 42
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], int, int]:
    """
    Parity-32 Benchmark:
    Computes exact XOR / odd parity over 32 active binary variables, zero-padded to 64 bits.
    A notoriously challenging non-linear benchmark for continuous floating-point architectures.
    """
    rng = np.random.RandomState(seed)
    total_samples = num_train + num_test
    total_pos = total_samples // 2
    total_neg = total_samples - total_pos

    seen_all: Set[Tuple[int, ...]] = set()

    pos_samples = []
    neg_samples = []

    while len(pos_samples) < total_pos or len(neg_samples) < total_neg:
        bits = rng.randint(0, 2, size=active_bits).astype(np.float32)
        parity = int(np.sum(bits)) % 2

        padded = np.zeros(reg_len, dtype=np.float32)
        padded[:active_bits] = bits
        tup = tuple(padded.astype(int))

        if tup in seen_all:
            continue
        seen_all.add(tup)

        if parity == 1 and len(pos_samples) < total_pos:
            pos_samples.append(padded)
        elif parity == 0 and len(neg_samples) < total_neg:
            neg_samples.append(padded)

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
