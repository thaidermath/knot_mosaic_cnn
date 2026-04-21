#!/usr/bin/env python3
"""Plot training/validation metrics from a fixed epoch table."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    epochs = list(range(1, 21))
    train_loss = [
        6.7060, 6.0394, 5.5064, 5.0033, 4.5303,
        4.0393, 3.5803, 3.2051, 2.8399, 2.5033,
        2.2079, 1.9730, 1.7375, 1.5603, 1.3997,
        1.2115, 1.1037, 0.9888, 0.8802, 0.7912,
    ]
    train_acc = [
        0.0228, 0.0715, 0.1361, 0.2157, 0.2875,
        0.3668, 0.4319, 0.4861, 0.5426, 0.5905,
        0.6402, 0.6739, 0.7185, 0.7425, 0.7649,
        0.7989, 0.8150, 0.8341, 0.8529, 0.8664,
    ]
    val_loss = [
        6.8903, 6.7931, 6.6126, 6.4934, 6.3520,
        6.1458, 5.8974, 5.7368, 5.5191, 5.3419,
        5.1993, 5.0790, 4.9710, 4.8222, 4.6801,
        4.5955, 4.5502, 4.4071, 4.2608, 4.2293,
    ]
    val_acc = [
        0.0031, 0.0063, 0.0094, 0.0201, 0.0227,
        0.0334, 0.0472, 0.0560, 0.0780, 0.0868,
        0.0925, 0.1064, 0.1233, 0.1454, 0.1422,
        0.1693, 0.1542, 0.1756, 0.2052, 0.2127,
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, train_loss, marker="o", label="Train Loss")
    axes[0].plot(epochs, val_loss, marker="o", label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss vs Epoch")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, marker="o", label="Train Acc")
    axes[1].plot(epochs, val_acc, marker="o", label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy vs Epoch")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle("ResNet18 Knot Training Log (20 epochs)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = Path("plots") / "resnet18_knot_training_log.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
