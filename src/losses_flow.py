import torch
import torch.nn.functional as F


def charbonnier(a, b, eps=1e-3):
    return torch.mean(torch.sqrt((a - b) ** 2 + eps * eps))


def _gray(x):
    return 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]


def sobel_mag_and_angle(x):
    g = _gray(x)

    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)

    ky = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)

    gx = F.conv2d(g, kx, padding=1)
    gy = F.conv2d(g, ky, padding=1)

    mag = torch.sqrt(gx * gx + gy * gy + 1e-8)
    ang = torch.atan2(gy, gx)
    return mag, ang


def edge_loss(pred, gt):
    pm, _ = sobel_mag_and_angle(pred)
    gm, _ = sobel_mag_and_angle(gt)
    return charbonnier(pm, gm)


def grad_orient_loss(pred, gt):
    _, pa = sobel_mag_and_angle(pred)
    _, ga = sobel_mag_and_angle(gt)
    return torch.mean(1.0 - torch.cos(pa - ga))


def flow_smoothness(flow):
    dx = torch.mean(torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1]))
    dy = torch.mean(torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]))
    return dx + dy


def ssim_loss(pred, gt):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(pred, 3, 1, 1)
    mu_y = F.avg_pool2d(gt, 3, 1, 1)

    sig_x = F.avg_pool2d(pred * pred, 3, 1, 1) - mu_x * mu_x
    sig_y = F.avg_pool2d(gt * gt, 3, 1, 1) - mu_y * mu_y
    sig_xy = F.avg_pool2d(pred * gt, 3, 1, 1) - mu_x * mu_y

    ssim_n = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
    ssim_d = (mu_x * mu_x + mu_y * mu_y + C1) * (sig_x + sig_y + C2)

    ssim = ssim_n / (ssim_d + 1e-8)
    return torch.mean(1.0 - ssim)
