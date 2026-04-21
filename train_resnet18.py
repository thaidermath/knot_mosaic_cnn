#!/usr/bin/env python3
"""Fine-tune ResNet18 on knot mosaic PNGs with class-balanced sampling."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms

from dataset_mosaic_images import build_image_dataloaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transfer-learn ResNet18 for knot classification")
    parser.add_argument("--data-root", default="dataset_split", help="Root directory containing train/val/test folders")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for optimizer")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze ResNet convolutional backbone and train only classifier head")
    parser.add_argument("--output", default="checkpoints/resnet18_knot.pt", help="Destination checkpoint path for best model")
    parser.add_argument("--evaluate-test", action="store_true", help="Run test-set evaluation after training")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing for cross-entropy loss")
    parser.add_argument("--log-every", type=int, default=10, help="Log metrics every N epochs")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_transforms(image_size: int = 224) -> dict[str, transforms.Compose]:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_tf = transforms.Compose([
        transforms.Resize(int(image_size * 1.1)),
        transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(image_size + 16),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return {"train": train_tf, "val": eval_tf, "test": eval_tf}


def build_model(num_classes: int, freeze_backbone: bool) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(dim=1) == targets).sum().item()
        total_examples += batch_size
    avg_loss = total_loss / max(total_examples, 1)
    avg_acc = total_correct / max(total_examples, 1)
    return avg_loss, avg_acc


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transforms_map = make_transforms()
    loaders = build_image_dataloaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transforms_map=transforms_map,
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_dataset, train_loader = loaders["train"]
    val_dataset, val_loader = loaders.get("val", (None, None))
    test_dataset, test_loader = loaders.get("test", (None, None))

    num_classes = len(train_dataset.label_names)
    model = build_model(num_classes, args.freeze_backbone)
    if args.freeze_backbone:
        for param in model.fc.parameters():
            param.requires_grad = True
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_acc = 0.0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = (float("nan"), float("nan"))
        if val_loader is not None:
            with torch.inference_mode():
                val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        elapsed = time.time() - start_time
        if epoch % max(args.log_every, 1) == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} time={elapsed:.1f}s"
            )
        if val_loader is not None and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": num_classes,
                    "label_names": train_dataset.label_names,
                    "args": vars(args),
                },
                output_path,
            )
            print(f"Saved checkpoint to {output_path}")

    if args.evaluate_test and test_loader is not None:
        checkpoint = torch.load(output_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        with torch.inference_mode():
            test_loss, test_acc = run_epoch(model, test_loader, criterion, None, device)
        print(f"Test results | loss={test_loss:.4f} acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
