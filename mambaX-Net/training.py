"""
This script contains the training loop for the model. It includes functions for training and evaluating the model, as well as saving checkpoints and logging metrics.

Input:
    --unet: path to the pretrained UNet model
    --data: path to the dataset 
    --output: path to the output folder

Author: Pierre-Louis Benveniste
"""
import argparse
import os
import json
import time
from loguru import logger
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from mambaxnet import load_nnunet_weights, MambaXNet
from load_dataset import get_dataloaders

 
# ──────────────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────────────
 
class DiceLoss(nn.Module):
    """Soft Dice loss for binary / multi-class segmentation.
    
    Works on logits; applies softmax internally.
    Expects:
        preds  – (B, C, *spatial)  raw logits
        targets – (B, *spatial)    integer class labels
    """
    def __init__(self, n_classes: int, smooth: float = 1e-5):
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth
 
    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(preds, dim=1)                          # (B, C, *)
        # One-hot encode targets → (B, C, *)
        targets_oh = torch.zeros_like(probs)
        targets_oh.scatter_(1, targets.unsqueeze(1).long(), 1.0)
 
        # Flatten spatial dims
        probs_flat = probs.view(probs.shape[0], probs.shape[1], -1)  # (B, C, N)
        tgt_flat   = targets_oh.view(*probs_flat.shape)
 
        intersection = (probs_flat * tgt_flat).sum(-1)               # (B, C)
        union        = probs_flat.sum(-1) + tgt_flat.sum(-1)         # (B, C)
 
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        # Mean over classes and batch; skip background class 0
        return 1.0 - dice[:, 1:].mean()
 
 
class CombinedLoss(nn.Module):
    """Dice + Cross-Entropy, equally weighted."""
    def __init__(self, n_classes: int):
        super().__init__()
        self.dice = DiceLoss(n_classes)
        self.ce   = nn.CrossEntropyLoss()
 
    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dice(preds, targets) + self.ce(preds, targets.long())
 


# ──────────────────────────────────────────────────────────────────────────────
# One epoch helpers
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model: nn.Module,
                    loader,
                    optimizer: optim.Optimizer,
                    criterion: nn.Module,
                    device: torch.device,
                    n_classes: int,
                    logger: logger,
                    epoch: int,
                    scaler: torch.cuda.amp.GradScaler | None) -> dict:
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    n_batches    = len(loader)

    for batch_idx, batch in enumerate(loader):
        images  = batch["image"].to(device, non_blocking=True)   # (B, C, *spatial)
        targets = batch["label"].to(device, non_blocking=True)   # (B, *spatial)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:                                    # mixed precision
            with torch.autocast(device_type=device.type):
                preds = model(images)
                loss  = criterion(preds, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(images)
            loss  = criterion(preds, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        with torch.no_grad():
            dice = compute_dice(preds.detach(), targets, n_classes)

        running_loss += loss.item()
        running_dice += dice

        if (batch_idx + 1) % max(1, n_batches // 5) == 0:
            logger.info(
                f"  Epoch {epoch:03d} | batch {batch_idx+1:>4d}/{n_batches} "
                f"| loss {loss.item():.4f} | dice {dice:.4f}"
            )

    return {
        "loss": running_loss / n_batches,
        "dice": running_dice / n_batches,
    }


@torch.no_grad()
def validate(model: nn.Module,
             loader,
             criterion: nn.Module,
             device: torch.device,
             n_classes: int) -> dict:
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    n_batches    = len(loader)

    for batch in loader:
        images  = batch["image"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)

        preds = model(images)
        loss  = criterion(preds, targets)
        dice  = compute_dice(preds, targets, n_classes)

        running_loss += loss.item()
        running_dice += dice

    return {
        "loss": running_loss / n_batches,
        "dice": running_dice / n_batches,
    }

# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────
 
def compute_dice(preds: torch.Tensor, targets: torch.Tensor,
                 n_classes: int, smooth: float = 1e-5) -> float:
    """Mean foreground Dice computed on hard predictions (argmax)."""
    pred_labels = preds.argmax(dim=1)                                # (B, *)
    dice_scores = []
    for cls in range(1, n_classes):                                  # skip background
        pred_c = (pred_labels == cls).float().view(-1)
        tgt_c  = (targets == cls).float().view(-1)
        inter  = (pred_c * tgt_c).sum()
        denom  = pred_c.sum() + tgt_c.sum()
        if denom == 0:
            continue                                                  # no GT & no pred → skip
        dice_scores.append(((2.0 * inter + smooth) / (denom + smooth)).item())
    return float(np.mean(dice_scores)) if dice_scores else 0.0
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(state: dict, path: str) -> None:
    torch.save(state, path)


def load_checkpoint(path: str, model: nn.Module,
                    optimizer: optim.Optimizer | None = None,
                    scheduler=None) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for the MambaXNet model.")
    parser.add_argument("--unet",     type=str, required=True,  help="Path to the pretrained UNet model.")
    parser.add_argument("--data",     type=str, required=True,  help="Path to the MSD dataset JSON.")
    parser.add_argument("--output",   type=str, required=True,  help="Path to the output folder.")
    # Training hyper-parameters (sensible defaults)
    parser.add_argument("--epochs",   type=int,   default=200,  help="Number of training epochs.")
    parser.add_argument("--lr",       type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument("--n_classes",type=int,   default=2,    help="Number of segmentation classes (incl. background).")
    parser.add_argument("--resume",   type=str,   default=None, help="Path to checkpoint to resume from.")
    parser.add_argument("--amp",      action="store_true",      help="Use automatic mixed precision (AMP).")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    unet_path   = args.unet
    data_path   = args.data
    output_path = args.output
    n_classes   = args.n_classes

    os.makedirs(output_path, exist_ok=True)
    
    # Build logger
    log_path = os.path.join(output_path, "training.log")
    logger.add(log_path, rotation="10 MB")
    logger.info("Starting MambaXNet training")

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Data ──────────────────────────────────────────────────────────────────
    logger.info("Loading dataset …")
    train_loader, val_loader, test_loader = get_dataloaders(json_path=data_path)
    logger.info(f"  Train batches : {len(train_loader)}")
    logger.info(f"  Val   batches : {len(val_loader)}")
    logger.info(f"  Test  batches : {len(test_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info("Initialising MambaXNet …")
    resencUnet = load_nnunet_weights(unet_path)
    model      = MambaXNet(n_channels=1, resenc_model=resencUnet, n_classes=n_classes)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Trainable parameters: {n_params:,}")

    # ── Loss / optimiser / scheduler ──────────────────────────────────────────
    criterion = CombinedLoss(n_classes=n_classes)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler    = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch   = 1
    best_val_dice = 0.0
    history       = {"train_loss": [], "train_dice": [],
                     "val_loss":   [], "val_dice":   []}

    if args.resume is not None:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        ckpt        = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_dice = ckpt.get("best_val_dice", 0.0)
        history     = ckpt.get("history", history)
        logger.info(f"  Resumed at epoch {start_epoch}, best val dice {best_val_dice:.4f}")

    # ── Training loop ─────────────────────────────────────────────────────────
    logger.info(f"Starting training for {args.epochs} epochs …")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.perf_counter()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, n_classes, logger, epoch, scaler,
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, n_classes)

        scheduler.step()

        elapsed = time.perf_counter() - t0

        # Log epoch summary
        logger.info(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train loss {train_metrics['loss']:.4f}  dice {train_metrics['dice']:.4f} | "
            f"Val   loss {val_metrics['loss']:.4f}  dice {val_metrics['dice']:.4f} | "
            f"LR {scheduler.get_last_lr()[0]:.2e} | "
            f"{elapsed:.1f}s"
        )

        # Update history
        history["train_loss"].append(train_metrics["loss"])
        history["train_dice"].append(train_metrics["dice"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_dice"].append(val_metrics["dice"])

        # ── Checkpoint: best model ────────────────────────────────────────────
        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            best_ckpt_path = os.path.join(output_path, "best_model.pth")
            save_checkpoint(
                {
                    "epoch":          epoch,
                    "model_state":    model.state_dict(),
                    "optimizer_state":optimizer.state_dict(),
                    "scheduler_state":scheduler.state_dict(),
                    "best_val_dice":  best_val_dice,
                    "history":        history,
                },
                best_ckpt_path,
            )
            logger.info(f"  ✓ New best val dice {best_val_dice:.4f} → saved to {best_ckpt_path}")

        # ── Checkpoint: latest ────────────────────────────────────────────────
        save_checkpoint(
            {
                "epoch":          epoch,
                "model_state":    model.state_dict(),
                "optimizer_state":optimizer.state_dict(),
                "scheduler_state":scheduler.state_dict(),
                "best_val_dice":  best_val_dice,
                "history":        history,
            },
            os.path.join(output_path, "latest_checkpoint.pth"),
        )

    # ── Save training history ─────────────────────────────────────────────────
    history_path = os.path.join(output_path, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to {history_path}")
    logger.info(f"Training complete. Best val dice: {best_val_dice:.4f}")


if __name__ == "__main__":
    main()