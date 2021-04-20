import torch
import torch.nn.functional as F
from torch import nn, einsum

class Decoder(nn.Module):
    def __init__(
            self,
            upsampler_scale_factor
        ):
            super().__init__()

            self.upsampler = Upsampler(upsampler_scale_factor=upsampler_scale_factor)

    def forward(self, x1, x2):
        x1 = self.upsampler(x1)
        x = torch.cat((x1, x2), 1)
        return x

class Upsampler(nn.Module):
    def __init__(
        self,
        upsampler_scale_factor
        ):
        super().__init__()
        self.upsampler = nn.UpsamplingBilinear2d(scale_factor=upsampler_scale_factor)

    def forward(self, x):
        x = self.upsampler(x)
        return x

class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)

