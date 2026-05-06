import torch
import torch.nn as nn


class ShapeExtractorModule(nn.Module):
    def __init__(self, in_channels=3, out_channels=32):
        super(ShapeExtractorModule, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, m_prev):
        return self.layers(m_prev)