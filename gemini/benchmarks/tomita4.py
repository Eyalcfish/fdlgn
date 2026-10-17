import numpy as np
import torch
from typing import Tuple, Set

def contains_three_consecutive_zeros(bits: np.ndarray, seq_len: int) -> bool:
    for i in range(seq_len - 2):
        if bits[i] == 0 and bits[i+1] == 0 and bits[i+2] == 0:
            return True
    return False

def generate_valid_tomita4(rng: np.random.RandomState, seq_len: int) -> np.ndarray:
    """Generates a random binary string of length seq_len that does NOT contain '000'."""
    bits = np.empty(seq_len, dtype=np.float32)
    trailing_zeros = 0
    for i in range(seq_len):
        if trailing_zeros == 2:
            # Must emit 1 to avoid '000'
            b = 1
        else:
            b = rng.randint(0, 2)
        bits[i] = b
        if b == 0:
            trailing_zeros += 1
        else:
            trailing_zeros = 0
    return bits

def get_tomita4_dataset(
    num_train: int = 4000,
    num_test: int = 1000,
    seq_len: int = 32,
    reg_len: int = 64,
    seed: int = 42
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], int, int]:
    """
    Tomita Grammar 4 Benchmark:
    Recognizes formal language L4 = { w in {0, 1}* | '000' is NOT a substring of w }.
    Zero-padded to 64-bit registers. 50/50 class balance with zero data leakage.
    """
    rng = np.random.RandomState(seed)
    total_samples = num_train + num_test
    total_pos = total_samples // 2
    total_neg = total_samples - total_pos

    seen_all: Set[Tuple[int, ...]] = set()

    # Positive samples: strictly satisfies grammar (no '000')
    pos_samples = []
    while len(pos_samples) < total_pos:
        s = generate_valid_tomita4(rng, seq_len)
        padded = np.zeros(reg_len, dtype=np.float32)
        padded[:seq_len] = s
        tup = tuple(padded.astype(int))
        if tup not in seen_all:
            seen_all.add(tup)
            pos_samples.append(padded)

    # Negative samples: violates grammar (contains at least one '000')
    neg_samples = []
    while len(neg_samples) < total_neg:
        s = rng.randint(0, 2, size=seq_len).astype(np.float32)
        if contains_three_consecutive_zeros(s, seq_len):
            padded = np.zeros(reg_len, dtype=np.float32)
            padded[:seq_len] = s
            tup = tuple(padded.astype(int))
            if tup not in seen_all:
                seen_all.add(tup)
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
