import torch
import torch.nn as nn
import torch.nn.functional as F


def gn(ch):
    return nn.GroupNorm(8, ch)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            gn(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            gn(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


def add_coords(x):
    B, C, H, W = x.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype),
        torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    coords = torch.stack([xx, yy], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
    return torch.cat([x, coords], dim=1)


class UNetFlowMulti(nn.Module):
    def __init__(self, base=48, max_flow=0.12):
        super().__init__()
        self.max_flow = max_flow

        self.e1 = ConvBlock(5, base)
        self.e2 = ConvBlock(base, base * 2, stride=2)
        self.e3 = ConvBlock(base * 2, base * 4, stride=2)
        self.e4 = ConvBlock(base * 4, base * 6, stride=2)
        self.b = ConvBlock(base * 6, base * 8, stride=2)

        self.d4 = ConvBlock(base * 8 + base * 6, base * 6)
        self.d3 = ConvBlock(base * 6 + base * 4, base * 4)
        self.d2 = ConvBlock(base * 4 + base * 2, base * 2)
        self.d1 = ConvBlock(base * 2 + base, base)

        self.h4 = nn.Conv2d(base * 6, 2, 3, padding=1)
        self.h3 = nn.Conv2d(base * 4, 2, 3, padding=1)
        self.h2 = nn.Conv2d(base * 2, 2, 3, padding=1)
        self.h1 = nn.Conv2d(base, 2, 3, padding=1)

    def up(self, x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        x = add_coords(x)

        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        b = self.b(e4)

        d4 = self.up(b, e4)
        d4 = self.d4(torch.cat([d4, e4], dim=1))

        d3 = self.up(d4, e3)
        d3 = self.d3(torch.cat([d3, e3], dim=1))

        d2 = self.up(d3, e2)
        d2 = self.d2(torch.cat([d2, e2], dim=1))

        d1 = self.up(d2, e1)
        d1 = self.d1(torch.cat([d1, e1], dim=1))

        f4 = torch.tanh(self.h4(d4)) * self.max_flow
        f3 = torch.tanh(self.h3(d3)) * self.max_flow
        f2 = torch.tanh(self.h2(d2)) * self.max_flow
        f1 = torch.tanh(self.h1(d1)) * self.max_flow

        return f1, f2, f3, f4
