# modify from https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
import einops


class LayerNorm(nn.Module):
    '''
    channels_last (default) : [b h w c]
    channels_first: [b c h w]
    '''
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format

        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":

            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]

            return x



class Block_Conv(nn.Module):
    '''
        (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    '''
    def __init__(self, in_channel, drop_path=0.0, layer_scale_init_value=1e-6):
        super(Block_Conv, self).__init__()
        # DWConv
        self.dwconv = nn.Conv2d(in_channel, in_channel, kernel_size=7, padding=3, groups=in_channel, bias=1)
        # layernorm
        self.norm = LayerNorm(in_channel, eps=1e-6, data_format="channels_first")
        # 1x1 conv
        self.pwconv1 = nn.Conv2d(in_channel, 4 * in_channel, 1, 1, 0, 1,bias=False)
        # gelu
        self.act = nn.GELU()
        # 1x1 conv
        self.pwconv2 = nn.Conv2d(in_channel * 4, in_channel, 1, 1, 0, 1,bias=False)

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((in_channel)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        '''

        :param x: [b c h w]
        :return: x': [b c h w]
        '''
        res = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        #print(f'x shape: {x.shape}')
        #print(f'gamma shape: {self.gamma.shape}')
        x = einops.rearrange(x, "b c h w -> b h w c")
        if self.gamma is not None:
            x = self.gamma * x

        x = einops.rearrange(x, "b h w c -> b c h w")
        x = res + self.drop_path(x)

        return x


class Block_Linear(nn.Module):
    '''
        (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    '''

    def __init__(self, in_channel, drop_path=0.0, layer_scale_init_value=1e-6):
        super(Block_Linear, self).__init__()
        # DWConv
        self.dwconv = nn.Conv2d(in_channel, in_channel, kernel_size=7, padding=3, groups=in_channel, bias=1)
        # layernorm
        self.norm = LayerNorm(in_channel, eps=1e-6, data_format="channels_last")
        # 1x1 conv
        self.fc1 = nn.Linear(in_channel, 4 * in_channel)
        # gelu
        self.act = nn.GELU()
        # 1x1 conv
        self.fc2 = nn.Linear(in_channel * 4, in_channel)

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((in_channel)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        '''

        :param x: [b c h w]
        :return: x': [b c h w]
        '''
        res = x
        x = self.dwconv(x)
        x = einops.rearrange(x, "b c h w -> b h w c")
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        #print(f'x shape: {x.shape}')
        #print(f'gamma shape: {self.gamma.shape}')
        if self.gamma is not None:
            x = self.gamma * x
        x = einops.rearrange(x, "b h w c -> b c h w")

        x = res + self.drop_path(x)

        return x

class ConvNext(nn.Module):
    '''
    in_channel: num of img channel
    num_classes=1000: out head
    depths=[3, 3, 9, 3]: num of blocks at each stage
    dims=[96, 192, 384, 768]: feature dim at each stage
    drop_path_rate=0.0: stochastic depth rate
    layer_scale_init_value=1e-6: init value for layer scale
    head_init_scale=1: Init scaling value for classifier weights and biases
    '''
    def __init__(self, in_channel=3, num_classes=1000, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.0, layer_scale_init_value=1e-6, head_init_scale=1.,):
        super(ConvNext, self).__init__()
        self.downsample = nn.ModuleList()

        stem = nn.Sequential(
            nn.Conv2d(in_channel, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )

            self.downsample.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rate = [x.item() for x in torch.linspace(0, drop_path_rate, sum((depths)))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block_Linear(in_channel=dims[i], drop_path=dp_rate[cur + j], layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )

            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)


    def _init_weights(self, m):
        if isinstance(m , (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample[i](x)
            x = self.stages[i](x)

        return self.norm(x.mean([-2, -1]))

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x

def convnext_tiny(**kwargs):
    model = ConvNext(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224)
    model = convnext_tiny(num_classes=5)
    y = model(x)
    print(y.shape)

