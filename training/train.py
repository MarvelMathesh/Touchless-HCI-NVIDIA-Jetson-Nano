#!/usr/bin/env python3
"""
Standalone training script for GestureNet.

Usage::

    # Train from collected gesture data
    python -m training.train

    # With custom options
    python -m training.train --data-dir data/gestures --epochs 100 --lr 0.001

    # Export to ONNX + TensorRT after training
    python -m training.train --export

After training, the following files are saved to models/weights/:
    - gesture_net.pth     (PyTorch checkpoint)
    - gesture_net.onnx    (ONNX model for TensorRT conversion)
    - training_log.json   (training metrics)

Compatible with Python 3.6+, PyTorch >= 1.8.
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, random_split
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.error("PyTorch is required for training. See requirements.txt for install instructions.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train GestureNet classifier")
    parser.add_argument("--data-dir", default="data/gestures",
                        help="Root directory with per-gesture landmark data")
    parser.add_argument("--output-dir", default="models/weights",
                        help="Directory to save trained model")
    parser.add_argument("--epochs", type=int, default=80,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay (L2 regularization)")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Fraction of data for validation")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--no-augmented", action="store_true",
                        help="Exclude augmented samples from training")
    parser.add_argument("--export", action="store_true",
                        help="Export to ONNX after training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch, return (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * features.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    """Validate, return (avg_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * features.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / max(total, 1), correct / max(total, 1)


def compute_confusion_matrix(model, loader, device, num_classes):
    """Compute confusion matrix for detailed analysis."""
    model.eval()
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            outputs = model(features)
            _, predicted = outputs.max(1)

            for true_label, pred_label in zip(labels.numpy(), predicted.cpu().numpy()):
                matrix[true_label][pred_label] += 1

    return matrix


def print_confusion_matrix(matrix, class_names):
    """Pretty-print confusion matrix with per-class metrics."""
    num_classes = len(class_names)

    # Header
    header = "%-14s" % "True \\ Pred"
    for name in class_names:
        header += " %6s" % name[:6]
    header += "  Recall"
    logger.info(header)
    logger.info("-" * len(header))

    # Rows
    for i in range(num_classes):
        row = "%-14s" % class_names[i][:14]
        row_total = matrix[i].sum()
        for j in range(num_classes):
            row += " %6d" % matrix[i][j]
        recall = matrix[i][i] / max(row_total, 1)
        row += "  %.3f" % recall
        logger.info(row)

    # Precision row
    logger.info("-" * len(header))
    prec_row = "%-14s" % "Precision"
    for j in range(num_classes):
        col_total = matrix[:, j].sum()
        prec = matrix[j][j] / max(col_total, 1)
        prec_row += " %6.3f" % prec
    logger.info(prec_row)


def main():
    if not TORCH_AVAILABLE:
        sys.exit(1)

    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training device: %s", device)

    # --- Load dataset ---
    from training.dataset import GestureDataset
    dataset = GestureDataset(
        data_dir=args.data_dir,
        include_augmented=not args.no_augmented,
    )

    if len(dataset) == 0:
        logger.error("No training data found in %s", args.data_dir)
        logger.error("Collect data first: python -m data.collector.dataset_collector")
        sys.exit(1)

    if len(dataset) < 50:
        logger.warning("Very small dataset (%d samples). Consider collecting more data.", len(dataset))

    # --- Train/val split ---
    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    logger.info("Train: %d samples, Val: %d samples", train_size, val_size)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    # --- Create model ---
    from models.gesture_net import GestureNet
    model = GestureNet(
        input_dim=81,
        num_classes=dataset.num_classes,
    ).to(device)

    logger.info("GestureNet: %d parameters",
                sum(p.numel() for p in model.parameters()))

    # --- Loss + optimizer ---
    class_weights = dataset.class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True,
    )

    # --- Training loop ---
    os.makedirs(args.output_dir, exist_ok=True)
    best_val_acc = 0.0
    best_val_loss = float("inf")
    no_improve_count = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    checkpoint_path = os.path.join(args.output_dir, "gesture_net.pth")
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("Starting training: %d epochs, batch_size=%d, lr=%.4f",
                args.epochs, args.batch_size, args.lr)
    logger.info("=" * 60)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        improved = ""
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            no_improve_count = 0
            improved = " *"

            # Save best checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "num_classes": dataset.num_classes,
                "input_dim": 81,
                "class_names": dataset.class_names,
            }, checkpoint_path)
        else:
            no_improve_count += 1

        if epoch % 5 == 0 or epoch <= 3 or improved:
            logger.info(
                "Epoch %3d/%d | Train: loss=%.4f acc=%.3f | Val: loss=%.4f acc=%.3f%s",
                epoch, args.epochs, train_loss, train_acc, val_loss, val_acc, improved,
            )

        # Early stopping
        if no_improve_count >= args.patience:
            logger.info("Early stopping at epoch %d (patience=%d)", epoch, args.patience)
            break

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("Training complete in %.1f seconds", elapsed)
    logger.info("Best validation accuracy: %.4f", best_val_acc)
    logger.info("Model saved to: %s", checkpoint_path)

    # --- Load best model for final evaluation ---
    model = GestureNet.load_checkpoint(checkpoint_path, device=str(device))

    # --- Confusion matrix ---
    full_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    cm = compute_confusion_matrix(model, full_loader, device, dataset.num_classes)
    logger.info("\nConfusion Matrix (full dataset):")
    print_confusion_matrix(cm, dataset.class_names)

    # --- Save training log ---
    log_path = os.path.join(args.output_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump({
            "best_val_acc": best_val_acc,
            "best_val_loss": best_val_loss,
            "epochs_trained": len(history["train_loss"]),
            "total_samples": len(dataset),
            "train_samples": train_size,
            "val_samples": val_size,
            "training_time_sec": elapsed,
            "history": history,
            "class_names": dataset.class_names,
            "confusion_matrix": cm.tolist(),
        }, f, indent=2)
    logger.info("Training log saved to: %s", log_path)

    # --- Optional ONNX export ---
    if args.export:
        onnx_path = os.path.join(args.output_dir, "gesture_net.onnx")
        model.export_onnx(onnx_path)
        logger.info("ONNX model exported to: %s", onnx_path)

        # Try TensorRT conversion
        try:
            from training.export_tensorrt import convert_onnx_to_tensorrt
            engine_path = os.path.join(args.output_dir, "gesture_net.engine")
            convert_onnx_to_tensorrt(onnx_path, engine_path)
        except Exception as e:
            logger.warning("TensorRT conversion skipped: %s", e)
            logger.info("Run on Jetson: python -m training.export_tensorrt %s", onnx_path)

    logger.info("=" * 60)
    logger.info("NEXT STEPS:")
    logger.info("  1. If not exported: python -m training.train --export")
    logger.info("  2. Convert to TensorRT: python -m training.export_tensorrt")
    logger.info("  3. System will auto-detect the model on next run")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
