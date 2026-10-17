# FDLGN Benchmarking Suite & Architecture Research Log

## 1. Overview & Purpose

This directory (`gemini/`) and companion document (`gemini.md`) host a comprehensive, modular benchmarking suite comparing diverse neural and logic gate architectures on fundamental Boolean, sequence, and regular-language benchmarks.

As requested:
- All custom reference implementations of the FDLGN algorithm have been **removed**.
- The core codebase in [`src/`](file:///c:/Users/fishministrator/Projects/fdlgn/src/) remains completely **untouched**.
- The suite evaluates external baselines and architectures first, establishing rigorous empirical performance baselines before connecting FDLGN.

---

## 2. Models in the Benchmarking Suite

The suite evaluates 6 distinct architectures across both floating-point and discrete/relaxed logic paradigms:

### 2.1 Conventional Neural Networks
1. **MLP (Multi-Layer Perceptron)**:
   - Deep feedforward neural network with LayerNorm and GELU non-linearities.
   - Evaluates general continuous representation learning.
2. **CNN (1D Convolutional Neural Network)**:
   - 1D sliding convolution with batch normalization, GELU, and max/adaptive pooling.
   - Inductive bias: translation equivariance and local spatial pattern matching.
3. **Transformer (Self-Attention Sequence Model)**:
   - Multi-Head Self-Attention (MHA) over tokenized binary register sequences with sinusoidal positional encodings and layer normalization.
   - Inductive bias: all-to-all token interaction and permutation-aware relational attention.

### 2.2 Differentiable Logic Gate Networks
4. **Basic DLGN (Petersen et al., NeurIPS 2022)**:
   - 2-input logic gates with 16-way categorical softmax distributions over all canonical 2-input Boolean truth tables.
   - Static random input wiring and `GroupSum` voting readout.
5. **LDLGN (Light DLGN / TT-LUT2, Rüttgers et al., ICLR 2026)**:
   - Input-Wise Parameterization (IWP): direct parameterization of the 4 minterms ($2^2=4$) via sigmoid multilinear relaxation, reducing parameter count by 75% relative to categorical DLGN.
6. **Conv Soft DenseSkip-1 LUT4 (from `argus-dlgn-research` / `dlgn_research` Exp 21/22)**:
   - **Stage 1 (Sliding Tree-Convolutions + OR-Pooling)**: Scans registers using 1D sliding windows of 4-input LUT trees ($K=8, S=1$), followed by continuous logical OR-pooling across positions ($1 - \prod (1 - a_{f,t})$).
   - **Stage 2 (DenseSkip-1 Multi-Hop Residual Backbone)**: 4-layer cascade where each gate samples 3 inputs from layer $l-1$ and 1 skip connection from all preceding layers $\{0 \dots l-1\}$ and the initial input.
   - **Stage 3 (GroupSum Readout)**: Partitions final gate activations into balanced class voting accumulators.
   - **Optimization**: Continuous multilinear relaxation (`mode="soft"`).

---

## 3. Standardized Benchmark Tasks

The suite tests models on 4 distinct problem domains (4,000 training samples, 1,000 test samples, balanced 50/50 splits, strict zero data leakage):

1. **Shifted Substring Matching**:
   - Detects an 8-bit needle (`10110010`) positioned at a uniformly random offset in a 64-bit register.
   - Positive samples contain the needle; negative samples are verified to not contain the needle.
   - Tests spatial translation invariance and exact multi-bit pattern matching.
2. **Tomita Grammar 4 (Formal Language / Automata)**:
   - Classifies strings from the regular language $L_4 = \{ w \in \{0, 1\}^* \mid \text{'000' is not a substring of } w \}$.
   - Tests sequential state tracking and long-range constraint satisfaction.
3. **Parity-32 (High-Frequency Non-Linearity)**:
   - Computes $\bigoplus_{i=0}^{31} x_i$ (32-bit odd parity zero-padded to 64 bits).
   - A notoriously hard problem for continuous gradient descent that produces catastrophic failure in standard MLPs.
4. **Majority-64 (Counting & Threshold Logic)**:
   - Computes $\mathbb{I}\left(\sum_{i=0}^{63} x_i \ge 32\right)$ across 64 binary inputs.
   - Tests spatial integration, accumulator scaling, and threshold logic.

---

## 4. Benchmarking Suite Directory Layout (`gemini/`)

```
gemini/
├── benchmarks/
│   ├── __init__.py               # Unified benchmark loader (load_benchmark)
│   ├── shifted_substring.py      # Shifted Substring generator
│   ├── tomita4.py                # Tomita Grammar 4 generator
│   ├── parity32.py               # Parity-32 generator
│   └── majority64.py             # Majority-64 generator
├── models/
│   ├── __init__.py               # Model registry & factory (create_model)
│   ├── base.py                   # Base module, GroupSum readout, parameter counters
│   ├── mlp.py                    # Multi-Layer Perceptron
│   ├── cnn.py                    # 1D Convolutional Neural Network
│   ├── transformer.py            # Sequence Transformer with Self-Attention
│   ├── dlgn.py                   # Basic DLGN (Petersen et al. 2022)
│   ├── ldlgn.py                  # Light DLGN / TT-LUT2 (Rüttgers et al. 2026)
│   └── conv_denseskip_lut4.py    # Conv Soft DenseSkip-1 LUT4 (Argus DLGN Research)
├── runner.py                     # Standardized training loop, speed & metric evaluator
├── run_suite.py                  # Master CLI runner for full benchmark matrix
├── run_suite.bat                 # Windows GPU launcher (auto-detects CUDA venv)
├── results.json                  # Output JSON logs of benchmark metrics
└── benchmark_summary.md          # Formatted Markdown report
```

---

## 5. Empirical Baseline Benchmark Results (15 Epochs on GPU)

The suite was executed across all 6 models and 4 benchmarks on the NVIDIA GeForce RTX 3060:

| Model | Benchmark | Params | Train Acc | Test Acc (Peak) | Train Time | Throughput |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **MLP** | shifted_substring | 25,602 | 92.12% | 59.90% (59.90%) | 3.53s | 376,393 /s |
| **CNN** | shifted_substring | 27,362 | 100.00% | **100.00% (100.00%)** | 5.05s | 296,473 /s |
| **Transformer** | shifted_substring | 17,282 | 50.02% | 50.00% (51.90%) | 9.29s | 108,650 /s |
| **DLGN** | shifted_substring | 8,192 | 50.00% | 50.00% (50.00%) | 9.09s | 109,593 /s |
| **LDLGN** | shifted_substring | 2,048 | 54.73% | 52.40% (53.20%) | 8.55s | 166,693 /s |
| **Conv Soft DenseSkip-1 LUT4** | shifted_substring | 9,728 | 65.15% | 53.90% (54.30%) | 21.04s | 52,428 /s |
| **MLP** | tomita4 | 25,602 | 98.83% | 91.70% (92.50%) | 2.68s | 1,149,198 /s |
| **CNN** | tomita4 | 27,362 | 100.00% | **100.00% (100.00%)** | 3.57s | 410,249 /s |
| **Transformer** | tomita4 | 17,282 | 74.60% | 77.00% (77.00%) | 8.46s | 126,005 /s |
| **DLGN** | tomita4 | 8,192 | 69.53% | 70.90% (70.90%) | 8.61s | 142,396 /s |
| **LDLGN** | tomita4 | 2,048 | 74.33% | 76.50% (76.50%) | 7.89s | 74,291 /s |
| **Conv Soft DenseSkip-1 LUT4** | tomita4 | 9,728 | 82.38% | 81.60% (81.60%) | 22.14s | 44,597 /s |
| **MLP** | parity32 | 25,602 | 78.55% | 52.60% (53.90%) | 3.16s | 934,419 /s |
| **CNN** | parity32 | 27,362 | 64.45% | 49.50% (50.80%) | 4.59s | 356,852 /s |
| **Transformer** | parity32 | 17,282 | 50.00% | 50.00% (50.00%) | 9.63s | 122,977 /s |
| **DLGN** | parity32 | 8,192 | 50.00% | 50.00% (50.00%) | 9.97s | 128,159 /s |
| **LDLGN** | parity32 | 2,048 | 50.00% | 50.00% (50.00%) | 8.94s | 157,775 /s |
| **Conv Soft DenseSkip-1 LUT4** | parity32 | 9,728 | 55.58% | 50.50% (52.90%) | 23.28s | 30,643 /s |
| **MLP** | majority64 | 25,602 | 96.40% | 80.20% (85.60%) | 3.31s | 565,533 /s |
| **CNN** | majority64 | 27,362 | 99.62% | 98.20% (98.90%) | 4.49s | 373,659 /s |
| **Transformer** | majority64 | 17,282 | 100.00% | **100.00% (100.00%)** | 9.52s | 105,711 /s |
| **DLGN** | majority64 | 8,192 | 86.50% | 83.10% (83.10%) | 9.51s | 142,421 /s |
| **LDLGN** | majority64 | 2,048 | 89.17% | 85.30% (85.40%) | 9.11s | 132,702 /s |
| **Conv Soft DenseSkip-1 LUT4** | majority64 | 9,728 | 96.22% | 92.60% (92.60%) | 22.84s | 47,035 /s |

### Key Benchmark Insights:
1. **Parity-32 Demonstrates the Continuous Gradient Bottleneck**:
   Every continuous gradient-based architecture (MLP, CNN, Transformer, DLGN, LDLGN, Conv-LUT4) fails completely on Parity-32 (~50% chance accuracy). Continuous relaxation surfaces are virtually flat or riddled with orthogonal local minima for high-frequency XOR parity. This benchmark directly highlights the promise of **Fully Discrete Logic Gate Networks (FDLGN)**, which operates directly on Boolean algebraic reductions.
2. **Translation Equivariance on Shifted Substring**:
   The CNN achieves 100% test accuracy on Shifted Substring due to sliding convolutional inductive bias, while flat DLGNs and MLPs struggle to detect the needle when displaced across register offsets.
3. **Formal Grammar Tracking (Tomita 4)**:
   CNN achieves 100% and MLP reaches 91.70%, with Conv Soft DenseSkip-1 LUT4 reaching 81.60% and Light DLGN reaching 76.50%.

---

## 6. How to Run the Benchmark Suite

To run all models across all benchmarks on GPU:

```powershell
cd gemini
.\run_suite.bat
```

To run a specific subset of models or benchmarks:
```powershell
cd gemini
.\run_suite.bat --models cnn conv_soft_denseskip_lut4 mlp --benchmarks shifted_substring tomita4 --epochs 20
```

---

## 7. Integration Interface for FDLGN

When your native FDLGN training algorithm is ready to be connected to the suite:
1. Wrap the FDLGN model with the standard `forward(x) -> logits` / `predict(x)` signature.
2. Register it in [`gemini/models/__init__.py`](file:///c:/Users/fishministrator/Projects/fdlgn/gemini/models/__init__.py) under `MODEL_REGISTRY["fdlgn"]`.
3. Add native discrete inference latency benchmarking to [`gemini/runner.py`](file:///c:/Users/fishministrator/Projects/fdlgn/gemini/runner.py).
