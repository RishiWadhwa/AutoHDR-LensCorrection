import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.ReLU(inplace=True),
    )

class UNetFlow(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = conv_block(3, 32)
        self.enc2 = conv_block(32, 64)
        self.enc3 = conv_block(64, 128)
        self.enc4 = conv_block(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.up3 = conv_block(256+128, 128)
        self.up2 = conv_block(128+64, 64)
        self.up1 = conv_block(64+32, 32)

        self.final = nn.Conv2d(32, 2, 1)
        
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d3 = F.interpolate(e4, scale_factor=2)
        d3 = self.up3(torch.cat([d3, e3], dim=1))

        d2 = F.interpolate(d3, scale_factor=2)
        d2 = self.up2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(d2, scale_factor=2)
        d1 = self.up1(torch.cat([d1, e1], dim=1))

        flow = self.final(d1)
        return flow