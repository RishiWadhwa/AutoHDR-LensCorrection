import os
import zipfile
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm

from dataset import list_test_images
from model_flow import UNetFlowMulti
from warp_flow import warp_with_flow

# ==========================
# SETTINGS
# ==========================

TEST_ROOT = "/kaggle/input/competitions/automatic-lens-correction/test-originals"
MODEL_PATH = "/kaggle/working/best_flow_unet.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = (640, 640)

print("device:", DEVICE)

# ==========================
# LOAD MODEL
# ==========================

model = UNetFlowMulti(base=48, max_flow=0.12).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ==========================
# INFERENCE
# ==========================

def to_tensor(img):
    img = img.resize(IMG_SIZE, Image.BICUBIC)
    arr = np.array(img).astype("float32") / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

def to_image(t):
    t = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    t = np.clip(t, 0, 1)
    return Image.fromarray((t * 255).astype("uint8"))

output_dir = "/kaggle/working/test_corrected_flow"
os.makedirs(output_dir, exist_ok=True)

test_images = list_test_images(TEST_ROOT)

with torch.no_grad():
    for path in tqdm(test_images):
        img = Image.open(path).convert("RGB")
        inp = to_tensor(img).to(DEVICE)

        f1, _, _, _ = model(inp)
        out = warp_with_flow(inp, f1)

        out_img = to_image(out)
        filename = os.path.basename(path)
        out_img.save(os.path.join(output_dir, filename))

# ==========================
# ZIP
# ==========================

zip_path = "/kaggle/working/test_corrected_flow.zip"

with zipfile.ZipFile(zip_path, "w") as z:
    for fname in os.listdir(output_dir):
        z.write(
            os.path.join(output_dir, fname),
            arcname=fname,
        )

print("✅ Submission ZIP created:", zip_path)
