import os
import copy
import torch
from torch import nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple

import einops


class PatchEmbed(nn.Module):
    '''
        x: [b c h w] -> [b embed_dim h/stride, w/stride]
    '''
    def __init__(self, patch_size=16, stride=16, padding=0, in_channels=3, embed_dim=768, norm_layer=None):
        super(PatchEmbed, self).__init__()
        patch_size=to_2tuple(patch_size)
        stride=to_2tuple(stride)
        padding=to_2tuple(padding)

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        #print(x.shape)
        x = self.norm(x)

        return x


class LayerNormChannel(nn.Module):
    '''
        layer norm for channel dim
        x: [b c h w] -> [b c h w]
    '''
    def __init__(self, num_channels, eps=1e-05):
        super(LayerNormChannel, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x + self.bias.unsqueeze(-1).unsqueeze(-1)

        return x


class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels, **kwargs):
        super(GroupNorm, self).__init__(1, num_channels, **kwargs)


class FFN(nn.Module):
    '''
    1x1 Conv implement
    x: [b c h w] -> [b c h w]
    '''
    def __init__(self, in_channels, hidden_dim=None, out_channels=None, act_layer=nn.GELU, drop=0.0):
        super(FFN, self).__init__()
        out_channels = out_channels or in_channels
        hidden_dim = hidden_dim or in_channels
        self.fc1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class FeedForward(nn.Module):
    '''
    1x1 Conv implement
    x: [b c h w] -> [b c h w]
    '''
    def __init__(self, in_channels, hidden_dim=None, out_channels=None, act_layer=nn.GELU, drop=0.0):
        super(FeedForward, self).__init__()
        out_channels = out_channels or in_channels
        hidden_dim = hidden_dim or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, out_channels)
        self.drop = nn.Dropout(drop)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):

        return self.pool(x) - x


class PoolFormerBlock(nn.Module):
    '''
        dim: embedding dim
        pool_size: pooling size
        mlp_ratio: mlp expansion ratio
        act_layer: activation
        norm_layer: normalization
        drop: dropout rate
        drop path: Stochastic Depth,

        use_layer_scale, --layer_scale_init_value: LayerScale,

    '''
    def __init__(self, embed_dim, pool_size=3, mlp_ratio=4, act_layer=nn.GELU, norm_layer=GroupNorm, drop=0.0, drop_path=0.0, use_layer_scale=True, layer_scale_init_value=1e-5):
        super(PoolFormerBlock, self).__init__()

        self.norm1 = norm_layer(embed_dim)
        self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = norm_layer(embed_dim)
        hidden_dim = int(mlp_ratio * embed_dim)
        self.ffn = FFN(in_channels=embed_dim, hidden_dim=hidden_dim, act_layer=act_layer, drop=drop)

        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale1 = nn.Parameter(layer_scale_init_value * torch.ones((embed_dim)), requires_grad=True)
            self.layer_scale2 = nn.Parameter(layer_scale_init_value * torch.ones((embed_dim)), requires_grad=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        '''

        :param x: [b c h w]
        :return: x' [b c h w]
        '''
        res = x
        if self.use_layer_scale:
            x = self.norm1(x)
            x = self.token_mixer(x) * self.layer_scale1.unsqueeze(-1).unsqueeze(-1)
            x = res + self.drop_path(x)

            res = x

            x = self.norm2(x)
            x = self.ffn(x) * self.layer_scale2.unsqueeze(-1).unsqueeze(-1)
            x = res + self.drop_path(x)

            return x

        else:
            x = self.norm1(x)
            x = self.token_mixer(x)
            x = res + self.drop_path(x)

            res = x

            x = self.norm2(x)
            x = self.ffn(x)
            x = res + self.drop_path(x)

            return x


def Blocks(dim, index, layers, pool_size, mlp_ratio=4, act_layer=nn.GELU, norm_layer=GroupNorm, drop=0.0, drop_path=0.0, use_layer_scale=True, layer_scale_init_value=1e-5):
    """
    poolformerblock for one stage
    """
    blocks=[]
    '''
        def __init__(self, embed_dim, pool_size=3, mlp_ratio=4, act_layer=nn.GELU, norm_layer=GroupNorm, drop=0.0, drop_path=0.0, use_layer_scale=True, layer_scale_init_value=1e-5):

    '''
    for idx in range(layers[index]):
        dp_ratio = drop_path * (idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(PoolFormerBlock(
            embed_dim=dim,
            pool_size=pool_size,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop=drop,
            drop_path=drop_path,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value
        ))

        blocks = nn.Sequential(*blocks)

        return blocks


class PoolFormer(nn.Module):
    def __init__(self,
                 layers=None,
                 embed_dims=None,
                 mlp_ratios=None,
                 downsamples=None,
                 pool_size=3,
                 norm_layer=GroupNorm,
                 act_layer=nn.GELU,
                 num_classes=1000,
                 in_patch_size=7, in_stride=4, in_padding=2,
                 down_patch_size=3, down_stride=2, down_padding=1,
                 drop=0.0, drop_path=0.0,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 fork_feat=None):
        '''

        :param layers:  [x, x, x, x], number of blocks for the 4 stages
        :param embed_dims:
        :param mlp_ratios:
        :param downsamples: flags to apply downsampling or not
        :param pool_size:
        :param norm_layer:
        :param act_layer:
        :param num_classes:
        :param in_patch_size: patch embedding for the input image
        :param in_stride: ...
        :param in_padding: ...
        :param down_patch_size:
        :param down_stride:
        :param down_padding:
        :param drop:
        :param drop_path:
        :param use_layer_scale:
        :param layer_scale_init_value:
        :param fork_feat:
        '''
        super(PoolFormer, self).__init__()


        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat

        self.patch_embed = PatchEmbed(
            patch_size=in_patch_size, stride=in_stride, padding=in_padding, in_channels=3, embed_dim=embed_dims[0]
        )


        '''
        def Blocks(dim, index, layers, pool_size, mlp_ratio=4, act_layer=nn.GELU, norm_layer=GroupNorm, drop=0.0, drop_path=0.0, use_layer_scale=True, layer_scale_init_value=1e-5):

        '''
        network = []
        for i in range(len(layers)):
            stage = Blocks(
                dim=embed_dims[i],
                index=i,
                layers=layers,
                pool_size=pool_size,
                mlp_ratio=mlp_ratios[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                drop=drop,
                drop_path=drop_path,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value
            )
            network.append(stage)

            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                # downsample between two stages
                network.append(
                    PatchEmbed(patch_size=down_patch_size, stride=down_stride, padding=down_padding, in_channels=embed_dims[i], embed_dim=embed_dims[i+1])
                )

        self.network = nn.ModuleList(network)

        self.norm = norm_layer(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    # TODO: more elegant way
                    """For RetinaNet, `start_level=1`. The first norm layer will not used.
                    cmd: `FORK_LAST3=1 python -m torch.distributed.launch ...`
                    """
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_token(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)

            if self.fork_feat:
                return outs

        return x

    def forward(self, x):
        x = self.forward_embeddings(x)
        x = self.forward_token(x)

        if self.fork_feat:
            return x

        x = self.norm(x)
        cls_out = self.head(x.mean([-2, -1]))
        return cls_out



@register_model
def poolformer_s12(pretrained=False, **kwargs):
    """
    PoolFormer-S12 model, Params: 12M
    --layers: [x,x,x,x], numbers of layers for the four stages
    --embed_dims, --mlp_ratios:
        embedding dims and mlp ratios for the four stages
    --downsamples: flags to apply downsampling or not in four blocks
    """
    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = PoolFormer(
        layers, embed_dims=embed_dims,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        **kwargs)
    return model

if __name__ == "__main__":
    x = torch.randn(1, 3, 256, 256)
    #x = einops.rearrange(x, 'b c h w -> b h w c')
    backbone = poolformer_s12(num_classes=5)
    y = backbone(x)
    print(y.shape)

