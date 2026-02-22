import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import LensPairDataset
from model_flow import UNetFlowMulti
from warp_flow import warp_with_flow
from losses_flow import (
    charbonnier,
    edge_loss,
    grad_orient_loss,
    flow_smoothness,
    ssim_loss,
)

# ==========================
# SETTINGS
# ==========================

TRAIN_ROOT = "/kaggle/input/competitions/automatic-lens-correction/lens-correction-train-cleaned"
IMG_SIZE = (640, 640)
BATCH_SIZE = 6
EPOCHS = 3
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("device:", DEVICE)

# ==========================
# DATA
# ==========================

full_ds = LensPairDataset(TRAIN_ROOT, resize=IMG_SIZE, train=True)
val_size = int(0.05 * len(full_ds))
train_size = len(full_ds) - val_size
train_ds, val_ds = random_split(full_ds, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ==========================
# MODEL
# ==========================

model = UNetFlowMulti(base=48, max_flow=0.12).to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

# ==========================
# TRAIN LOOP
# ==========================

best_val = 999
start = time.time()

for epoch in range(EPOCHS):
    model.train()
    for step, batch in enumerate(train_loader):

        distorted = batch["distorted"].to(DEVICE)
        gt = batch["corrected"].to(DEVICE)

        f1, f2, f3, f4 = model(distorted)

        pred = warp_with_flow(distorted, f1)

        loss_pix = charbonnier(pred, gt)
        loss_edge = edge_loss(pred, gt)
        loss_ang = grad_orient_loss(pred, gt)
        loss_ssim = ssim_loss(pred, gt)
        loss_smooth = flow_smoothness(f1)

        loss = (
            loss_pix
            + 0.5 * loss_edge
            + 0.3 * loss_ang
            + 0.2 * loss_ssim
            + 0.01 * loss_smooth
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 100 == 0:
            mins = (time.time() - start) / 60
            print(
                f"ep {epoch} step {step} "
                f"loss {loss.item():.4f} "
                f"| pix {loss_pix.item():.4f} "
                f"edge {loss_edge.item():.4f} "
                f"ang {loss_ang.item():.4f} "
                f"ssim {loss_ssim.item():.4f} "
                f"({mins:.1f} min)"
            )

    # ======================
    # VALIDATION
    # ======================

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            distorted = batch["distorted"].to(DEVICE)
            gt = batch["corrected"].to(DEVICE)

            f1, _, _, _ = model(distorted)
            pred = warp_with_flow(distorted, f1)
            val_loss += charbonnier(pred, gt).item()

    val_loss /= len(val_loader)
    print("VAL LOSS:", val_loss)

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "/kaggle/working/best_flow_unet.pt")
        print("✅ Saved new best model.")

print("Training done.")
