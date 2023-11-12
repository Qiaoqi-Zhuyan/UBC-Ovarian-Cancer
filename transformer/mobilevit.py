import torch
from torch import nn
import einops

x = torch.randn(1, 3, 224, 224)

class conv_1x1_bn(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(conv_1x1_bn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.SiLU()
        )

    def forward(self, x):
        return self.conv(x)


class conv_nxn_bn(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, kernel_size=3):
        super(conv_nxn_bn, self).__init__()
        self.conv_nxn = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, 0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.SiLU()
        )

    def forward(self, x):
        return self.conv_nxn(x)


class Feedforward(nn.Module):
    def __init__(self, dim, hidden_dim, droupout=0.0):
        super(Feedforward, self).__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(droupout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(droupout)
        )

    def forward(self, x):
        return self.feed_forward(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, head_dim=64, dropout=0.0):
        super(Attention, self).__init__()
        inner_dim = head_dim * heads
        project_out = not (heads == 1 and head_dim == dim)

        self.heads = heads
        self.scale = head_dim ** -0.5

        self.attn = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b p n (h d) -> b p h n d', h = self.heads), qkv)
        dots = torch.matual(q, k.transpose(-1, -2)) * self.scale

        attn = self.attn(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = einops(out, 'b h n d -> b n (h d)')
        return self.out(out)

class Transformer(nn.Module):
    def __init__(self, in_channel, depth, heads, head_dim, hidden_dim, drop=0.0):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([

            ]))
