"""
METHOD 3: UNet++ with EfficientNet-B4 backbone
Run: python method3_unetpp/train.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from utils.dataset import BurnDataset
from utils.losses import DiceFocalLoss
from utils.metrics import compute_metrics, print_metrics

# ── Config ────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE  = 320
BATCH     = 2
EPOCHS    = 80
LR        = 3e-4
CKPT_OUT  = os.path.join(BASE, 'checkpoints', 'unetpp', 'best.pth')
TRAIN_IMG = os.path.join(BASE, 'dataset', 'train', 'images')
TRAIN_MSK = os.path.join(BASE, 'dataset', 'train', 'masks')
TEST_IMG  = os.path.join(BASE, 'dataset', 'test',  'images')
TEST_MSK  = os.path.join(BASE, 'dataset', 'test',  'masks')


def dice_score(pred_logits, masks):
    pred = (torch.sigmoid(pred_logits) > 0.5).float()
    inter = (pred * masks).sum()
    return (2 * inter / (pred.sum() + masks.sum() + 1e-6)).item()


def validate(model, loader):
    model.eval()
    scores = []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            scores.append(dice_score(model(imgs), masks))
    return np.mean(scores)


def train():
    print(f"Device: {DEVICE}")

    # Data
    train_ds = BurnDataset(TRAIN_IMG, TRAIN_MSK, IMG_SIZE, augment=True)
    test_ds  = BurnDataset(TEST_IMG,  TEST_MSK,  IMG_SIZE, augment=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=2, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=1,     shuffle=False, num_workers=1)

    # Model
    model = smp.UnetPlusPlus(
        encoder_name='efficientnet-b4',
        encoder_weights='imagenet',
        in_channels=3,
        classes=1,
        activation=None,
    ).to(DEVICE)

    criterion = DiceFocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

    best_dice = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for imgs, masks in train_dl:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            loss = criterion(model(imgs), masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        val_dice = validate(model, test_dl)
        print(f"Epoch {epoch:3d}/{EPOCHS} | Loss: {total_loss/len(train_dl):.4f} | Val Dice: {val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), CKPT_OUT)
            print(f"  -> Saved best model (Dice: {best_dice:.4f})")

    print(f"\nTraining complete. Best Dice: {best_dice:.4f}")
    print(f"Checkpoint: {CKPT_OUT}")


if __name__ == '__main__':
    train()
