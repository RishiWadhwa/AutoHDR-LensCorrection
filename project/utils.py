import torch
import torch.nn.functional as F

def warp(image, flow):
    B, C, H, W = image.size()

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H),
        torch.linspace(-1, 1, W),
        indexing='ij'
    )

    grid = torch.stack((grid_x, grid_y), 2).to(image.device)
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)

    flow_norm = torch.zeros_like(flow)
    flow_norm[:, 0, :, :] = flow[:, 0, :, :] / (W/2)
    flow_norm[:, 1, :, :] = flow[:, 1, :, :] / (H/2)

    flow_norm = flow_norm.permute(0, 2, 3, 1)
    new_grid = grid + flow_norm

    return F.grid_sample(image, new_grid, align_corners=True)