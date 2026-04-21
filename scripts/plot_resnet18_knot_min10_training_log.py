#!/usr/bin/env python3
"""Plot training/validation metrics for ResNet18 knot_min10 table."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    epochs = list(range(1, 21))
    train_loss = [
        5.9945, 5.6500, 5.3270, 4.9573, 4.5616,
        4.1817, 3.8223, 3.4976, 3.1846, 2.9313,
        2.6857, 2.4648, 2.2846, 2.1292, 2.0151,
        1.8798, 1.7673, 1.6851, 1.6137, 1.5584,
    ]
    train_acc = [
        0.0181, 0.0474, 0.0935, 0.1504, 0.2322,
        0.3089, 0.3833, 0.4621, 0.5243, 0.5928,
        0.6455, 0.7059, 0.7449, 0.7793, 0.8049,
        0.8398, 0.8646, 0.8807, 0.8973, 0.9116,
    ]
    val_loss = [
        6.0964, 6.0462, 5.8043, 5.5247, 5.3934,
        5.1578, 4.9129, 4.6879, 4.4488, 4.2588,
        4.0578, 4.1302, 3.8532, 3.8037, 3.6793,
        3.5500, 3.4409, 3.3920, 3.3386, 3.1814,
    ]
    val_acc = [
        0.0054, 0.0141, 0.0173, 0.0346, 0.0498,
        0.0747, 0.0974, 0.1158, 0.1526, 0.2002,
        0.2219, 0.1948, 0.2435, 0.2532, 0.2803,
        0.3203, 0.3312, 0.3734, 0.3864, 0.4242,
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

    fig.suptitle("ResNet18 Knot Min10 Training Log (20 epochs)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = Path("plots") / "resnet18_knot_min10_training_log.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
