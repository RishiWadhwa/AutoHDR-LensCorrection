import os
import glob
import random
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset


def list_test_images(root):
    return sorted(glob.glob(os.path.join(root, "*.jpg")))


class LensPairDataset(Dataset):
    """
    Loads:
    *_original.jpg  -> distorted input
    *_generated.jpg -> corrected target
    """

    def __init__(self, root, resize=(640, 640), train=True):
        self.root = root
        self.resize = resize
        self.train = train

        self.pairs = []
        originals = glob.glob(os.path.join(root, "*_original.jpg"))
        for o in originals:
            g = o.replace("_original.jpg", "_generated.jpg")
            if os.path.exists(g):
                self.pairs.append((o, g))
        self.pairs.sort()

    def __len__(self):
        return len(self.pairs)

    def _open(self, path):
        return Image.open(path).convert("RGB")

    def _pair_geom_aug(self, a, b):
        if random.random() < 0.5:
            a = a.transpose(Image.FLIP_LEFT_RIGHT)
            b = b.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() < 0.7:
            W, H = a.size
            scale = 0.7 + 0.3 * random.random()
            cw, ch = int(W * scale), int(H * scale)
            if cw >= 32 and ch >= 32:
                x0 = random.randint(0, max(0, W - cw))
                y0 = random.randint(0, max(0, H - ch))
                a = a.crop((x0, y0, x0 + cw, y0 + ch))
                b = b.crop((x0, y0, x0 + cw, y0 + ch))
        return a, b

    def _maybe_color(self, img):
        if random.random() < 0.25:
            img = ImageEnhance.Brightness(img).enhance(0.9 + 0.2 * random.random())
        if random.random() < 0.25:
            img = ImageEnhance.Contrast(img).enhance(0.9 + 0.2 * random.random())
        return img

    def _to_tensor(self, img):
        img = img.resize(self.resize, Image.BICUBIC)
        arr = np.array(img).astype("float32") / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return t

    def __getitem__(self, idx):
        o, g = self.pairs[idx]
        d = self._open(o)
        gt = self._open(g)

        if self.train:
            d, gt = self._pair_geom_aug(d, gt)
            d = self._maybe_color(d)

        return {
            "distorted": self._to_tensor(d),
            "corrected": self._to_tensor(gt),
        }
