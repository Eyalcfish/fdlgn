# FDLGN Research - Neural & Logic Architecture Benchmark Summary

Generated with device: `cuda` | Epochs: `15`

## Overall Results Table

| Model                      | Benchmark            | Params     | Train Acc    | Test Acc (Peak)    | Train Time   | Throughput       |
|----------------------------|----------------------|------------|--------------|--------------------|--------------|------------------|
| mlp                        | shifted_substring    | 25,602     | 92.12%       | 59.90% (59.90%)    | 3.53s        | 376,393/s        |
| cnn                        | shifted_substring    | 27,362     | 100.00%      | 100.00% (100.00%)  | 5.05s        | 296,473/s        |
| transformer                | shifted_substring    | 17,282     | 50.02%       | 50.00% (51.90%)    | 9.29s        | 108,650/s        |
| dlgn                       | shifted_substring    | 8,192      | 50.00%       | 50.00% (50.00%)    | 9.09s        | 109,593/s        |
| ldlgn                      | shifted_substring    | 2,048      | 54.73%       | 52.40% (53.20%)    | 8.55s        | 166,693/s        |
| conv_soft_denseskip_lut4   | shifted_substring    | 9,728      | 65.15%       | 53.90% (54.30%)    | 21.04s       | 52,428/s         |
| mlp                        | tomita4              | 25,602     | 98.83%       | 91.70% (92.50%)    | 2.68s        | 1,149,198/s      |
| cnn                        | tomita4              | 27,362     | 100.00%      | 100.00% (100.00%)  | 3.57s        | 410,249/s        |
| transformer                | tomita4              | 17,282     | 74.60%       | 77.00% (77.00%)    | 8.46s        | 126,005/s        |
| dlgn                       | tomita4              | 8,192      | 69.53%       | 70.90% (70.90%)    | 8.61s        | 142,396/s        |
| ldlgn                      | tomita4              | 2,048      | 74.33%       | 76.50% (76.50%)    | 7.89s        | 74,291/s         |
| conv_soft_denseskip_lut4   | tomita4              | 9,728      | 82.38%       | 81.60% (81.60%)    | 22.14s       | 44,597/s         |
| mlp                        | parity32             | 25,602     | 78.55%       | 52.60% (53.90%)    | 3.16s        | 934,419/s        |
| cnn                        | parity32             | 27,362     | 64.45%       | 49.50% (50.80%)    | 4.59s        | 356,852/s        |
| transformer                | parity32             | 17,282     | 50.00%       | 50.00% (50.00%)    | 9.63s        | 122,977/s        |
| dlgn                       | parity32             | 8,192      | 50.00%       | 50.00% (50.00%)    | 9.97s        | 128,159/s        |
| ldlgn                      | parity32             | 2,048      | 50.00%       | 50.00% (50.00%)    | 8.94s        | 157,775/s        |
| conv_soft_denseskip_lut4   | parity32             | 9,728      | 55.58%       | 50.50% (52.90%)    | 23.28s       | 30,643/s         |
| mlp                        | majority64           | 25,602     | 96.40%       | 80.20% (85.60%)    | 3.31s        | 565,533/s        |
| cnn                        | majority64           | 27,362     | 99.62%       | 98.20% (98.90%)    | 4.49s        | 373,659/s        |
| transformer                | majority64           | 17,282     | 100.00%      | 100.00% (100.00%)  | 9.52s        | 105,711/s        |
| dlgn                       | majority64           | 8,192      | 86.50%       | 83.10% (83.10%)    | 9.51s        | 142,421/s        |
| ldlgn                      | majority64           | 2,048      | 89.17%       | 85.30% (85.40%)    | 9.11s        | 132,702/s        |
| conv_soft_denseskip_lut4   | majority64           | 9,728      | 96.22%       | 92.60% (92.60%)    | 22.84s       | 47,035/s         |

## Architectures Evaluated
- **MLP**: Classical Multi-Layer Perceptron (LayerNorm + GELU activations)
- **CNN**: 1D Convolutional Neural Network (Conv1D + MaxPool + BatchNorm)
- **Transformer**: Sequence Self-Attention Transformer with positional encodings
- **Basic DLGN**: 16-operator categorical softmax logic gate network (Petersen et al., NeurIPS 2022)
- **LDLGN**: Light DLGN with 4-minterm Input-Wise Parameterization (Rüttgers et al., ICLR 2026)
- **Conv Soft DenseSkip-1 LUT4**: Tree-convolutional sliding window with OR-pooling and DenseSkip backbone (argus-dlgn-research Exp21/22)
