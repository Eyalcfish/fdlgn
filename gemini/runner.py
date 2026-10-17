import time
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from models import create_model, count_parameters
from benchmarks import load_benchmark

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """Evaluates accuracy and loss on a given dataloader."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x_b, y_b in loader:
            x_b = x_b.to(device)
            y_b = y_b.to(device)

            logits = model(x_b)
            loss = criterion(logits, y_b)

            total_loss += loss.item() * len(y_b)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == y_b).sum().item()
            total += len(y_b)

    avg_loss = total_loss / max(1, total)
    acc = 100.0 * correct / max(1, total)
    return acc, avg_loss

def benchmark_inference_speed(model: nn.Module, x_sample: torch.Tensor, device: torch.device, reps: int = 100) -> Tuple[float, float]:
    """Measures single-batch inference throughput and per-sample latency."""
    model.eval()
    x_sample = x_sample.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x_sample)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(reps):
            _ = model(x_sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_end = time.perf_counter()

    total_time = t_end - t_start
    total_samples = reps * len(x_sample)
    throughput = total_samples / max(1e-8, total_time)
    latency_us = (total_time / max(1, total_samples)) * 1e6

    return throughput, latency_us

def train_and_evaluate(
    model_name: str,
    benchmark_name: str,
    epochs: int = 20,
    batch_size: int = 64,
    lr: Optional[float] = None,
    seed: int = 42,
    num_train: int = 4000,
    num_test: int = 1000,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Standardized training and benchmark execution for a single (model, benchmark) pair.
    """
    if device is None:
        device = get_device()

    # Load dataset
    (X_tr, y_tr), (X_te, y_te), in_features, num_classes = load_benchmark(
        benchmark_name, num_train=num_train, num_test=num_test, seed=seed
    )

    train_dataset = TensorDataset(X_tr, y_tr)
    test_dataset = TensorDataset(X_te, y_te)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate model
    torch.manual_seed(seed)
    model = create_model(model_name, in_features=in_features, num_classes=num_classes).to(device)
    params = count_parameters(model)

    # Appropriate learning rate by model family
    if lr is None:
        if model_name in ["dlgn", "ldlgn", "conv_soft_denseskip_lut4"]:
            lr = 0.01
        elif model_name == "transformer":
            lr = 0.001
        else:
            lr = 0.003

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    if verbose:
        print(f"\n[{model_name.upper()}] on [{benchmark_name}] | Params: {params:,} | Device: {device} | LR: {lr}")

    t_train_start = time.perf_counter()
    best_test_acc = 0.0
    best_epoch = 0

    for ep in range(1, epochs + 1):
        model.train()
        for x_b, y_b in train_loader:
            x_b = x_b.to(device)
            y_b = y_b.to(device)

            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

        # Evaluate progress
        te_acc, te_loss = evaluate_model(model, test_loader, device)
        if te_acc > best_test_acc:
            best_test_acc = te_acc
            best_epoch = ep

        if verbose and (ep == 1 or ep % 5 == 0 or ep == epochs):
            tr_acc, tr_loss = evaluate_model(model, train_loader, device)
            print(f"  Epoch {ep:2d}/{epochs:2d} | Train Acc: {tr_acc:6.2f}% | Test Acc: {te_acc:6.2f}% | Loss: {te_loss:.4f}")

    t_train_end = time.perf_counter()
    train_time = t_train_end - t_train_start

    # Final evaluations
    final_tr_acc, final_tr_loss = evaluate_model(model, train_loader, device)
    final_te_acc, final_te_loss = evaluate_model(model, test_loader, device)

    # Inference speed benchmark (using test set slice)
    speed_x = X_te[:min(256, len(X_te))]
    throughput, latency_us = benchmark_inference_speed(model, speed_x, device)

    results = {
        "model": model_name,
        "benchmark": benchmark_name,
        "parameters": params,
        "train_acc": final_tr_acc,
        "test_acc": final_te_acc,
        "best_test_acc": best_test_acc,
        "best_epoch": best_epoch,
        "train_loss": final_tr_loss,
        "test_loss": final_te_loss,
        "training_time_sec": train_time,
        "throughput_samples_per_sec": throughput,
        "latency_us_per_sample": latency_us,
        "device": str(device)
    }

    if verbose:
        print(f"  -> Result: Test Acc = {final_te_acc:.2f}% (Peak {best_test_acc:.2f}%) | Time: {train_time:.2f}s | Speed: {throughput:,.0f} samples/s ({latency_us:.2f} us/sample)")

    return results
