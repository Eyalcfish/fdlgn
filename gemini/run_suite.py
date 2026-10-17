import os
import sys
import json
import argparse
from typing import List, Dict, Any
import torch

from models import MODEL_REGISTRY
from benchmarks import BENCHMARK_REGISTRY
from runner import train_and_evaluate, get_device

def format_table(results: List[Dict[str, Any]]) -> str:
    headers = ["Model", "Benchmark", "Params", "Train Acc", "Test Acc (Peak)", "Train Time", "Throughput"]
    col_widths = [26, 20, 10, 12, 18, 12, 16]

    def make_row(cols):
        return "| " + " | ".join(f"{str(c):<{w}}" for c, w in zip(cols, col_widths)) + " |"

    def make_sep():
        return "|-" + "-|-".join("-" * w for w in col_widths) + "-|"

    lines = [make_row(headers), make_sep()]
    for r in results:
        cols = [
            r["model"],
            r["benchmark"],
            f"{r['parameters']:,}",
            f"{r['train_acc']:.2f}%",
            f"{r['test_acc']:.2f}% ({r['best_test_acc']:.2f}%)",
            f"{r['training_time_sec']:.2f}s",
            f"{r['throughput_samples_per_sec']:,.0f}/s"
        ]
        lines.append(make_row(cols))

    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="FDLGN Gemini Benchmarking Suite - Comparing Diverse Neural & Logic Architectures")
    parser.add_argument("--models", nargs="+", default=["mlp", "cnn", "transformer", "dlgn", "ldlgn", "conv_soft_denseskip_lut4"],
                        choices=list(MODEL_REGISTRY.keys()), help="Models to benchmark")
    parser.add_argument("--benchmarks", nargs="+", default=["shifted_substring", "tomita4", "parity32", "majority64"],
                        choices=list(BENCHMARK_REGISTRY.keys()), help="Benchmarks to evaluate")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs per run")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--train-samples", type=int, default=4000, help="Number of training samples")
    parser.add_argument("--test-samples", type=int, default=1000, help="Number of test samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-json", type=str, default="results.json", help="Path to save output JSON results")
    args = parser.parse_args()

    device = get_device()
    print("================================================================================")
    print("         FDLGN Research - Comprehensive Architecture Benchmarking Suite")
    print("================================================================================")
    print(f"Device:           {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Selected Models:  {args.models}")
    print(f"Benchmarks:       {args.benchmarks}")
    print(f"Epochs per run:   {args.epochs}")
    print(f"Dataset Size:     {args.train_samples} Train / {args.test_samples} Test")
    print("================================================================================")

    all_results: List[Dict[str, Any]] = []

    for bench in args.benchmarks:
        print(f"\n>>>>>>>>>>>>>>>>>>>> STARTING BENCHMARK: {bench.upper()} <<<<<<<<<<<<<<<<<<<<")
        for model in args.models:
            try:
                res = train_and_evaluate(
                    model_name=model,
                    benchmark_name=bench,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    seed=args.seed,
                    num_train=args.train_samples,
                    num_test=args.test_samples,
                    device=device,
                    verbose=True
                )
                all_results.append(res)
            except Exception as e:
                print(f"  [ERROR] Model '{model}' failed on benchmark '{bench}': {e}")

    # Output formatting
    print("\n\n================================================================================")
    print("                           FINAL BENCHMARK RESULTS")
    print("================================================================================")
    summary_table = format_table(all_results)
    print(summary_table)

    # Save to JSON
    out_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(out_dir, args.output_json)
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to: {json_path}")

    # Save to Markdown
    md_path = os.path.join(out_dir, "benchmark_summary.md")
    with open(md_path, "w") as f:
        f.write("# FDLGN Research - Neural & Logic Architecture Benchmark Summary\n\n")
        f.write(f"Generated with device: `{device}` | Epochs: `{args.epochs}`\n\n")
        f.write("## Overall Results Table\n\n")
        f.write(summary_table)
        f.write("\n\n## Architectures Evaluated\n")
        f.write("- **MLP**: Classical Multi-Layer Perceptron (LayerNorm + GELU activations)\n")
        f.write("- **CNN**: 1D Convolutional Neural Network (Conv1D + MaxPool + BatchNorm)\n")
        f.write("- **Transformer**: Sequence Self-Attention Transformer with positional encodings\n")
        f.write("- **Basic DLGN**: 16-operator categorical softmax logic gate network (Petersen et al., NeurIPS 2022)\n")
        f.write("- **LDLGN**: Light DLGN with 4-minterm Input-Wise Parameterization (Rüttgers et al., ICLR 2026)\n")
        f.write("- **Conv Soft DenseSkip-1 LUT4**: Tree-convolutional sliding window with OR-pooling and DenseSkip backbone (argus-dlgn-research Exp21/22)\n")
    print(f"Summary markdown saved to: {md_path}")

if __name__ == "__main__":
    main()
