# modify from: https://github.com/sail-sg/poolformer/blob/main/models/poolformer.py
# modify from: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple
import einops


class LayerNormChannel(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: [B, C, H, W] -> output: [B, C, H, W]
    """

    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x


class PatchEmbed(nn.Module):
    """
    input: [B, C, H, W] -> output: [B, C, H/stride, W/stride]
    """

    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class PatchEmbed_swin(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        y = self.proj(x)
        print(y.shape)
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


class Stem(nn.Module):
    '''
        input: [B, C, H, W] -> output: [B, C, H/stride, W/stride]
        conv -> LayerNorm
    '''

    def __init__(self, in_channels=3, patch_size=4, stride=4, padding=0, embed_dim=96, norm_layer=nn.LayerNorm):
        super(Stem, self).__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = einops.rearrange(x, "b c h w -> b h w c")
        x = self.norm(x)
        x = einops.rearrange(x, "b h w c-> b c h w ")
        return x


class Downsample(nn.Module):
    '''
        input: [B, C, H, W] -> output: [B, C * 2, H // 2, W // 2]
    '''
    def __init__(self, in_channel, norm_layer=nn.LayerNorm, type="linear"):
        super(Downsample, self).__init__()
        self.norm = norm_layer(4 * in_channel)
        self.fc = nn.Linear(4 * in_channel, 2 * in_channel)
        self.conv = nn.Conv2d(in_channels=4 * in_channel, out_channels=2 * in_channel, kernel_size=1, stride=1)
        self.type = type

    def forward(self, x):
        _, _, H, W = x.shape

        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        assert type == "linear" or type == "conv", "worry type"

        x0 = x[:, :, 0::2, 0::2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2] # -> [B, C, H//2, W//2]

        x = torch.cat([x0, x1, x2, x3], 1)

        x = einops.rearrange(x, "b c h w -> b h w c")
        x = self.norm(x)

        if self.type == "linear":
            x = self.fc(x)
            return einops.rearrange(x, "b h w c -> b c h w")

        elif self.type == "conv":
            x = einops.rearrange(x, "b h w c -> b c h w")
            return self.conv(x)


class ConvFFN(nn.Module):
    '''
     input: [B C H W] ->
    '''
    def __init__(self, in_channel, out_channel, expan_ratio=4):
        super(ConvFFN, self).__init__()
        hidden_dim = int(expan_ratio * in_channel)
        self.pwconv1 = nn.Conv2d(in_channels=in_channel, out_channels=hidden_dim, kernel_size=1)
        self.pwconv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=out_channel, kernel_size=1)
        self.dwconv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, groups=in_channel)
        self.act_layer = nn.GELU()
        self.norm = nn.LayerNorm(in_channel)
        self.apply(self._init_wights)

    def _init_wights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.dwconv(x)
        x = einops.rearrange(x, "b c h w -> b h w c")
        x = self.norm(x)
        x = einops.rearrange(x, "b h w c -> b c h w")
        x = self.pwconv1(x)
        x = self.act_layer(x)
        x = self.pwconv2(x)

        return x


class ConvFormerBlock(nn.Module):
    def __init__(self, dim, expan_ration=4, norm_layer=nn.LayerNorm, drop=0.0, drop_path=0.0, use_layer_scale=True, layer_scale_init_value=1e-5):
        super(ConvFormerBlock, self).__init__()
        self.norm_layer = norm_layer
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3)
        self.act_layer = nn.GELU()

    def _attn_mixer(self,):
        pass

    def _conv_mixer(self, ):
        pass


    def forward(self, x):
        pass


class ConvFormer(nn.Module):
    pass







if __name__ == "__main__":
    x = torch.randn(1, 3, 256, 256)
    norm = PatchEmbed(patch_size=4, stride=4, embed_dim=96)
    norm_swin = PatchEmbed_swin(patch_size=4, embed_dim=96)
    embed = Stem()
    downsample = Downsample(3)
    ffn = ConvFFN(in_channel=3, out_channel=56)
    y = ffn(x)
    print(f"Block module: {y.shape}")


