import torch
from torch import nn
import einops

img = torch.randn(1, 3, 18, 18)

_, _, w, h = img.shape

resize_img = einops.rearrange(img, 'a b c d ->  a c b d')
print(resize_img)
print(resize_img.shape)

