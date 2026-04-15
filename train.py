import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from safetensors.torch import save_file

from dataset import ViolenceDataset
from model import ViolenceClassifier


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True, help="Path to dataset root")
    p.add_argument("--output", default="models/violence_classifier.safetensors")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for clips, labels in loader:
        clips, labels = clips.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(clips)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * clips.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += clips.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for clips, labels in loader:
        clips, labels = clips.to(device), labels.to(device)
        logits = model(clips)
        loss = criterion(logits, labels)
        total_loss += loss.item() * clips.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += clips.size(0)
    return total_loss / total, correct / total


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = ViolenceDataset(args.data_root, split="train",
                               num_frames=args.num_frames, img_size=args.img_size)
    val_ds   = ViolenceDataset(args.data_root, split="val",
                               num_frames=args.num_frames, img_size=args.img_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    model = ViolenceClassifier(num_frames=args.num_frames).to(device)

    # Phase 1: Freeze backbone for first few epochs
    for param in model.features.parameters():
        param.requires_grad = False
    # No update for backbone
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val_acc = 0.0
    UNFREEZE_EPOCH = 5  # unfreeze backbone after warmup

    for epoch in range(1, args.epochs + 1):
        if epoch == UNFREEZE_EPOCH:
            print("Unfreezing backbone…")
            for param in model.features.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.1)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - UNFREEZE_EPOCH
            )

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch [{epoch:02d}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            save_file(model.state_dict(), args.output)
            print(f"Saved best model (val_acc={val_acc:.4f}) -> {args.output}")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()