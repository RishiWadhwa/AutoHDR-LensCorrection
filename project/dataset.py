import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class DistortionDataset():
    def __init__(self, root_dir, file_list, size=512):
        self.root_dir = root_dir
        self.files = file_list
        self.size = size

        self.transform = T.Compose([
            T.Resize(size),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        distorted_path = os.path.join(self.root_dir, file)
        gt_path = distorted_path.replace("_original.jpg", "_generated.jpg")

        distorted = Image.open(distorted_path).convert("RGB")
        gt = Image.open(gt_path).convert("RGB")

        distorted = self.transform(distorted)
        gt = self.transform(gt)

        return distorted, gt