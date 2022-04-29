from numpy.core.fromnumeric import shape
import torch
from transformer_in_transformer import TNTUNet
import cv2
import numpy as np
from einops import rearrange

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def read_data(image_path):
    imgs = []
    img = rearrange(cv2.imread(image_path), 'w h c -> c w h')
    imgs.append(img)
    imgs = torch.from_numpy(np.array(imgs)).type(torch.FloatTensor)
    return imgs

# tnt = TNT(
#     image_size = 256,       # size of image
#     patch_dim = 512,        # dimension of patch token
#     pixel_dim = 24,         # dimension of pixel token
#     patch_size = 16,        # patch size
#     pixel_size = 4,         # pixel size
#     depth = 6,              # depth
#     num_classes = 1000,     # output number of classes
#     attn_dropout = 0.1,     # attention dropout
#     ff_dropout = 0.1        # feedforward dropout
# )

# # temp = torch.randn(1, 3, 256, 256)
# print("reading image ...")
# imgs = read_data('./data/test.png')
# print('predicting ...')
# raw = tnt(imgs) # (2, 1000)
# print(raw.shape)


# image_size_1 = 256
# image_size_2 = 64
# image_size_3 = 16
# patch_size = 4
# pixel_feature_dim=512
# class_num = 100

# tnt_4_first = TNT(
#     image_size = image_size_1,       # size of image
#     patch_dim = 512,        # dimension of patch token
#     pixel_dim = 24,         # dimension of pixel token
#     patch_size = patch_size,        # patch size
#     pixel_size = 2,         # pixel size
#     depth = 2,              # depth
#     pixel_feature_dim=3,
#     attn_dropout = 0.1,     # attention dropout
#     ff_dropout = 0.1        # feedforward dropout
# )

# tnt_4_second = TNT(
#     image_size = image_size_2,       # size of image
#     patch_dim = 512,        # dimension of patch token
#     pixel_dim = 24,         # dimension of pixel token
#     patch_size = patch_size,        # patch size
#     pixel_size = 2,         # pixel size
#     depth = 2,              # depth
#     pixel_feature_dim=pixel_feature_dim,
#     attn_dropout = 0.1,     # attention dropout
#     ff_dropout = 0.1        # feedforward dropout
# )

# tnt_4_third = TNT(
#     image_size = image_size_3,       # size of image
#     patch_dim = 512,        # dimension of patch token
#     pixel_dim = 24,         # dimension of pixel token
#     patch_size = patch_size,        # patch size
#     pixel_size = 2,         # pixel size
#     depth = 2,              # depth
#     pixel_feature_dim=pixel_feature_dim,
#     attn_dropout = 0.1,     # attention dropout
#     ff_dropout = 0.1        # feedforward dropout
# )

# conv2dReLU = Conv2dReLU(
#     in_channels=1024,
#     out_channels=pixel_feature_dim,
#     kernel_size=3,
#     padding=1
# )

# decoder = Decoder(
#     upsampler_scale_factor = 4
# )

# upsampler = Upsampler(
#     upsampler_scale_factor=4
#     )

# segmentation=Segmentation(in_channels=pixel_feature_dim, out_channels=class_num, kernel_size=3)





# print("reading image ...")
# imgs = read_data('./data/test.png')
# print("Data shape", imgs.shape)
# print('predicting layer 1 ...')
# output_1 = tnt_4_first(imgs)
# output_1 = rearrange(output_1, 'b (h w) d -> b d h w', h=int(output_1.shape[1]**0.5), w=int(output_1.shape[1]**0.5))
# print(output_1.shape)
# print('predicting layer 2 ...')
# output_2 = tnt_4_second(output_1)
# output_2 = rearrange(output_2, 'b (h w) d -> b d h w', h=int(output_2.shape[1]**0.5), w=int(output_2.shape[1]**0.5))
# print(output_2.shape)
# print('predicting layer 3 ...')
# output_3 = tnt_4_third(output_2)
# output_3 = rearrange(output_3, 'b (h w) d -> b d h w', h=int(output_3.shape[1]**0.5), w=int(output_3.shape[1]**0.5))
# print(output_3.shape)


# print('decoding layer 1 ...')
# output_decoder_1 = decoder(output_3, output_2)
# print(output_decoder_1.shape)
# output_decoder_1 = conv2dReLU(output_decoder_1)
# print(output_decoder_1.shape)
# print('decoding layer 2 ...')
# output_decoder_2 = decoder(output_decoder_1, output_1)
# print(output_decoder_2.shape)
# output_decoder_2 = conv2dReLU(output_decoder_2)
# print(output_decoder_2.shape)
# print('decoding layer 3 ...')
# output_decoder_3 = upsampler(output_decoder_2)
# print(output_decoder_3.shape)
# print('segmentation layer 3 ...')
# final = segmentation(output_decoder_3)
# print(final.shape)


print("reading image ...")
imgs = read_data('./data/test.png')

tntunet = TNTUNet()
out = tntunet(imgs)
print(out.shape)

