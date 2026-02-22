import torch
from torch.utils.data import DataLoader
from model import UNetFlow
from dataset import DistortionDataset
from utils import warp
import glob
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

files = sorted([f for f in os.listdir("train") if f.endswith("_original.jpg")])
train_files = files[:20000]
val_files = files[20000:]

train_dataset = DistortionDataset("train", train_files)
val_dataset = DistortionDataset("train", val_files)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

model = UNetFlow().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    model.train()
    total_loss = 0

    for i, (x, y) in enumerate(train_loader):
        if i % 50 == 0:
            print(f"Epoch {epoch} | Batch {i}/{len(train_loader)}")
        x, y = x.to(device), y.to(device)

        flow = model(x)
        warped = warp(x, flow)

        loss = torch.nn.functional.l1_loss(warped, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch} Loss: {total_loss/len(train_loader)}")

torch.save(model.state_dict(), "model_day1.pth")