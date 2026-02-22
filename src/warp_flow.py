import torch
import torch.nn.functional as F


def make_base_grid(B, H, W, device, dtype):
    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device, dtype=dtype),
        torch.linspace(-1, 1, W, device=device, dtype=dtype),
        indexing="ij",
    )
    grid = torch.stack([xs, ys], dim=-1)
    return grid.unsqueeze(0).repeat(B, 1, 1, 1)


def warp_with_flow(img, flow, padding_mode="reflection"):
    B, C, H, W = img.shape
    base = make_base_grid(B, H, W, img.device, img.dtype)
    grid = base + flow.permute(0, 2, 3, 1).contiguous()

    return F.grid_sample(
        img,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )
