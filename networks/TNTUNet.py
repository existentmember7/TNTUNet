import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import cv2
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

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

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(val, divisor):
    return (val % divisor) == 0

def unfold_output_size(image_size, kernel_size, stride, padding):
    return int(((image_size - kernel_size + (2 * padding)) / stride) + 1)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, d, h = *x.shape, self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

# main class

class TNT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_dim,
        pixel_dim,
        patch_size,
        pixel_size,
        depth,
        pixel_feature_dim,
        # num_classes,
        heads = 8,
        dim_head = 64,
        ff_dropout = 0.,
        attn_dropout = 0.,
        unfold_args = None
    ):
        super().__init__()
        assert divisible_by(image_size, patch_size), 'image size must be divisible by patch size'
        assert divisible_by(patch_size, pixel_size), 'patch size must be divisible by pixel size for now'

        num_patch_tokens = (image_size // patch_size) ** 2

        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_tokens = nn.Parameter(torch.randn(num_patch_tokens + 1, patch_dim))

        unfold_args = default(unfold_args, (pixel_size, pixel_size, 0))
        unfold_args = (*unfold_args, 0) if len(unfold_args) == 2 else unfold_args
        kernel_size, stride, padding = unfold_args

        pixel_width = unfold_output_size(patch_size, kernel_size, stride, padding)
        num_pixels = pixel_width ** 2

        self.to_pixel_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> (b h w) c p1 p2', p1 = patch_size, p2 = patch_size),
            nn.Unfold(kernel_size = kernel_size, stride = stride, padding = padding),
            Rearrange('... c n -> ... n c'),
            nn.Linear(pixel_feature_dim * kernel_size ** 2, pixel_dim)
        )

        self.patch_pos_emb = nn.Parameter(torch.randn(num_patch_tokens + 1, patch_dim))
        self.pixel_pos_emb = nn.Parameter(torch.randn(num_pixels, pixel_dim))

        layers = nn.ModuleList([])
        for _ in range(depth):

            pixel_to_patch = nn.Sequential(
                nn.LayerNorm(pixel_dim),
                Rearrange('... n d -> ... (n d)'),
                nn.Linear(pixel_dim * num_pixels, patch_dim),
            )

            layers.append(nn.ModuleList([
                PreNorm(pixel_dim, Attention(dim = pixel_dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)),
                PreNorm(pixel_dim, FeedForward(dim = pixel_dim, dropout = ff_dropout)),
                pixel_to_patch,
                PreNorm(patch_dim, Attention(dim = patch_dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)),
                PreNorm(patch_dim, FeedForward(dim = patch_dim, dropout = ff_dropout)),
            ]))

        self.layers = layers


    def forward(self, x):
        b, _, h, w, patch_size, image_size = *x.shape, self.patch_size, self.image_size
        assert divisible_by(h, patch_size) and divisible_by(w, patch_size), f'height {h} and width {w} of input must be divisible by the patch size'

        num_patches_h = h // patch_size
        num_patches_w = w // patch_size
        n = num_patches_w * num_patches_h
        # print("num_patches_w: ", num_patches_w)
        # print("num_patches_h: ", num_patches_h)

        # print("x: ", x.shape)
        pixels = self.to_pixel_tokens(x)
        # print("pixels: ", pixels.shape)

        patches = repeat(self.patch_tokens[:(n + 1)], 'n d -> b n d', b = b)
        # print("patches:", patches.shape)

        patches += rearrange(self.patch_pos_emb[:(n + 1)], 'n d -> () n d')
        pixels += rearrange(self.pixel_pos_emb, 'n d -> () n d')

        for pixel_attn, pixel_ff, pixel_to_patch_residual, patch_attn, patch_ff in self.layers:

            pixels = pixel_attn(pixels) + pixels
            pixels = pixel_ff(pixels) + pixels

            # print("pixels:", pixels.shape)
            patches_residual = pixel_to_patch_residual(pixels)
            # print("patches_residual:", patches_residual.shape)

            patches_residual = rearrange(patches_residual, '(b h w) d -> b (h w) d', h = num_patches_h, w = num_patches_w)
            patches_residual = F.pad(patches_residual, (0, 0, 1, 0), value = 0) # cls token gets residual of 0
            # print("patches_residual: ", patches_residual.shape)
            patches = patches + patches_residual

            patches = patch_attn(patches) + patches
            patches = patch_ff(patches) + patches


        # cls_token = patches[:, 0]
        return patches[:,1:]

class Segmentation(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)

class MLP(nn.Sequential):

    def __init__(
        self,
        feature_dim,
        num_classes
    ):
        super().__init__()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.feature),
            nn.Linear(self.patch_dim, self.num_classes)
        )

class TNTUNet(nn.Module):
    def __init__(
        self,
        image_size = 256,
        patch_size = 4,
        pixel_feature_dim=512,
        class_num = 100,
        channels = 4
    ):
        super().__init__()

        self.tnt_4_first = TNT(
            image_size = image_size,       # size of image
            patch_dim = 512,        # dimension of patch token
            pixel_dim = 24,         # dimension of pixel token
            patch_size = patch_size,        # patch size
            pixel_size = 2,         # pixel size
            depth = 2,              # depth
            pixel_feature_dim=channels,
            attn_dropout = 0.1,     # attention dropout
            ff_dropout = 0.1        # feedforward dropout
        )


        self.tnt_4_second = TNT(
            image_size = int(image_size/patch_size),       # size of image
            patch_dim = 512,        # dimension of patch token
            pixel_dim = 24,         # dimension of pixel token
            patch_size = patch_size,        # patch size
            pixel_size = 2,         # pixel size
            depth = 2,              # depth
            pixel_feature_dim=pixel_feature_dim,
            attn_dropout = 0.1,     # attention dropout
            ff_dropout = 0.1        # feedforward dropout
        )

        self.tnt_4_third = TNT(
            image_size = int(image_size/(patch_size)**2),       # size of image
            patch_dim = 512,        # dimension of patch token
            pixel_dim = 24,         # dimension of pixel token
            patch_size = patch_size,        # patch size
            pixel_size = 2,         # pixel size
            depth = 2,              # depth
            pixel_feature_dim=pixel_feature_dim,
            attn_dropout = 0.1,     # attention dropout
            ff_dropout = 0.1        # feedforward dropout
        )

        self.conv2dReLU = Conv2dReLU(
            in_channels=1024,
            out_channels=pixel_feature_dim,
            kernel_size=3,
            padding=1
        )

        self.decoder = Decoder(
            upsampler_scale_factor=4
        )

        self.upsampler = Upsampler(
            upsampler_scale_factor=4
            )

        self.segmentation=Segmentation(in_channels=pixel_feature_dim, out_channels=class_num, kernel_size=3)

    def forward(self, x):
        output_1 = self.tnt_4_first(x)
        output_1 = rearrange(output_1, 'b (h w) d -> b d h w', h=int(output_1.shape[1]**0.5), w=int(output_1.shape[1]**0.5))
        # print(output_1.shape)
        # print('predicting layer 2 ...')
        output_2 = self.tnt_4_second(output_1)
        output_2 = rearrange(output_2, 'b (h w) d -> b d h w', h=int(output_2.shape[1]**0.5), w=int(output_2.shape[1]**0.5))
        # print(output_2.shape)
        # print('predicting layer 3 ...')
        output_3 = self.tnt_4_third(output_2)
        output_3 = rearrange(output_3, 'b (h w) d -> b d h w', h=int(output_3.shape[1]**0.5), w=int(output_3.shape[1]**0.5))
        # print(output_3.shape)


        # print('decoding layer 1 ...')
        output_decoder_1 = self.decoder(output_3, output_2)
        # print(output_decoder_1.shape)
        output_decoder_1 = self.conv2dReLU(output_decoder_1)
        # print(output_decoder_1.shape)
        # print('decoding layer 2 ...')
        output_decoder_2 = self.decoder(output_decoder_1, output_1)
        # print(output_decoder_2.shape)
        output_decoder_2 = self.conv2dReLU(output_decoder_2)
        # print(output_decoder_2.shape)
        # print('decoding layer 3 ...')
        output_decoder_3 = self.upsampler(output_decoder_2)
        # print(output_decoder_3.shape)
        # print('segmentation layer 3 ...')
        final_output = self.segmentation(output_decoder_3)
        # print(final_output.shape)
        return final_output
