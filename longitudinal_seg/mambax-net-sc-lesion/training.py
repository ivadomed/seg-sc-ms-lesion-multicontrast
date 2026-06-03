"""
Training loop for MambaXNet-SC-Lesion.

The model receives five inputs (image_M12, image_M0, sc_mask_M0, sc_mask_M12,
lesion_mask_M0) and predicts the lesion segmentation at M12.

Input:
    --unet   : path to the pretrained UNet model
    --data   : path to the dataset JSON
    --output : path to the output folder

Author: Pierre-Louis Benveniste
"""

import argparse
import os
import json
import time
from loguru import logger
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb

from mambaxnet_sc_lesion import load_nnunet_weights, MambaXNetSCLesion
from load_dataset import get_dataloaders


# ──────────────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    def __init__(self, n_classes: int, smooth: float = 1e-5):
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(preds, dim=1)
        targets_oh = torch.zeros_like(probs)
        targets_oh.scatter_(1, targets.unsqueeze(1).long(), 1.0)

        probs_flat = probs.view(probs.shape[0], probs.shape[1], -1)
        tgt_flat   = targets_oh.view(*probs_flat.shape)

        intersection = (probs_flat * tgt_flat).sum(-1)
        union        = probs_flat.sum(-1) + tgt_flat.sum(-1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice[:, 1:].mean()


class CombinedLoss(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.dice = DiceLoss(n_classes)
        self.ce   = nn.CrossEntropyLoss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dice(preds, targets) + self.ce(preds, targets.long())


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_dice(preds: torch.Tensor, targets: torch.Tensor,
                 n_classes: int, smooth: float = 1e-5) -> float:
    pred_labels = preds.argmax(dim=1)
    dice_scores = []
    for cls in range(1, n_classes):
        pred_c = (pred_labels == cls).float().view(-1)
        tgt_c  = (targets == cls).float().view(-1)
        inter  = (pred_c * tgt_c).sum()
        denom  = pred_c.sum() + tgt_c.sum()
        if denom == 0:
            return 1.0
        dice_scores.append(((2.0 * inter + smooth) / (denom + smooth)).item())
    return float(np.mean(dice_scores)) if dice_scores else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Forward pass
# ──────────────────────────────────────────────────────────────────────────────

def forward_pair(model: nn.Module, batch: dict, device: torch.device):
    """
    Unpack the five model inputs and the prediction target from a batch.

    Batch keys (all tensors of shape (B, 1, *spatial)):
        image1       – M0 image
        image2       – M12 image
        sc_mask1     – SC mask at M0
        sc_mask2     – SC mask at M12
        lesion_mask1 – lesion mask at M0
        lesion_mask2 – lesion mask at M12  (target)
    """
    image_M0        = batch["image1"].to(device)
    image_M12       = batch["image2"].to(device)
    sc_mask_M0      = batch["sc_mask1"].to(device)
    sc_mask_M12     = batch["sc_mask2"].to(device)
    lesion_mask_M0  = batch["lesion_mask1"].to(device)
    lesion_mask_M12 = batch["lesion_mask2"].to(device)

    targets = lesion_mask_M12.squeeze(1)   # (B, *spatial)
    preds   = model(image_M12, image_M0, sc_mask_M0, sc_mask_M12, lesion_mask_M0)
    return preds, targets


# ──────────────────────────────────────────────────────────────────────────────
# Train / validate
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, n_classes, epoch, global_step):
    model.train()
    total_loss = 0.0
    total_dice = 0.0

    for batch_idx, batch in enumerate(loader):
        optimizer.zero_grad()
        preds, targets = forward_pair(model, batch, device)
        loss = criterion(preds, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            dice = compute_dice(preds.detach(), targets, n_classes)

        total_loss += loss.item()
        total_dice += dice
        global_step += 1

        wandb.log({"train/batch_loss": loss.item(), "train/batch_dice": dice}, step=global_step)
        logger.info(
            f"  Epoch {epoch} | batch {batch_idx+1}/{len(loader)} "
            f"| loss {loss.item():.4f} | dice {dice:.4f}"
        )

    return total_loss / len(loader), total_dice / len(loader), global_step


@torch.no_grad()
def validate(model, loader, criterion, device, n_classes):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0

    for batch in loader:
        preds, targets = forward_pair(model, batch, device)
        total_loss += criterion(preds, targets).item()
        total_dice += compute_dice(preds, targets, n_classes)

    return total_loss / len(loader), total_dice / len(loader)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet",           type=str, required=True)
    parser.add_argument("--data",           type=str, required=True)
    parser.add_argument("--output",         type=str, required=True)
    parser.add_argument("--epochs",         type=int,   default=200)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--n_classes",      type=int,   default=2)
    parser.add_argument("--wandb_project",  type=str,   default="mambaxnet-sc-lesion")
    parser.add_argument("--wandb_run",      type=str,   default=None)
    parser.add_argument("--wandb_offline",  action="store_true")
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Freeze encoder weights (stem + all stages).")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    output_path = os.path.join(args.output, f"training_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(output_path, exist_ok=True)

    logger.add(os.path.join(output_path, "training.log"), rotation="10 MB")
    logger.info(f"Dataset : {args.data}")
    logger.info(f"Output  : {output_path}")

    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"
    wandb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args), dir=output_path)
    logger.info(f"W&B run: {wandb.run.name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    logger.info("Loading dataset …")
    train_loader, val_loader, _ = get_dataloaders(json_path=args.data, batch_size=1)
    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    logger.info("Initialising MambaXNetSCLesion …")
    resenc_model = load_nnunet_weights(args.unet)
    model = MambaXNetSCLesion(n_channels=1, resenc_model=resenc_model, n_classes=args.n_classes)
    model.to(device)

    if args.freeze_encoder:
        encoder_modules = [
            model.enc_stem,
            model.enc_stage0, model.enc_stage1, model.enc_stage2,
            model.enc_stage3, model.enc_stage4, model.enc_stage5,
        ]
        for m in encoder_modules:
            for p in m.parameters():
                p.requires_grad = False
        logger.info("Encoder frozen.")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_params:,}")

    criterion = CombinedLoss(n_classes=args.n_classes)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_dice = 0.0
    global_step   = 0
    history = {"train_loss": [], "train_dice": [], "val_loss": [], "val_dice": []}

    logger.info(f"Starting training for {args.epochs} epochs …")
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()

        train_loss, train_dice, global_step = train_one_epoch(
            model, train_loader, optimizer, criterion, device, args.n_classes, epoch, global_step
        )
        val_loss, val_dice = validate(
            model, val_loader, criterion, device, args.n_classes
        )

        elapsed = time.perf_counter() - t0
        logger.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss {train_loss:.4f} dice {train_dice:.4f} | "
            f"val loss {val_loss:.4f} dice {val_dice:.4f} | "
            f"{elapsed:.1f}s"
        )

        wandb.log({
            "epoch":            epoch,
            "train/epoch_loss": train_loss,
            "train/epoch_dice": train_dice,
            "val/loss":         val_loss,
            "val/dice":         val_dice,
            "epoch_time_s":     elapsed,
        }, step=global_step)

        history["train_loss"].append(train_loss)
        history["train_dice"].append(train_dice)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), os.path.join(output_path, "best_model.pth"))
            logger.info(f"  New best val dice: {best_val_dice:.4f}")
            wandb.log({"val/best_dice": best_val_dice}, step=global_step)

    history_path = os.path.join(output_path, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Done. Best val dice: {best_val_dice:.4f}")
    wandb.summary["best_val_dice"] = best_val_dice
    wandb.finish()


if __name__ == "__main__":
    main()
