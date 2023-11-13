'''
copy from https://github.com/chinhsuanwu/mobilevit-pytorch/blob/master/mobilevit.py
'''



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
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1):
        super(conv_nxn_bn, self).__init__()
        self.conv_nxn = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, 1, bias=False),
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
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attn(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)

        out = einops.rearrange(out, 'b p h n d -> b p n (h d)')
        return self.out(out)

class Transformer(nn.Module):
    def __init__(self, in_channel, depth, heads, head_dim, hidden_dim, attn_drop=0.0, ffn_drop=0.0):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        """
        class Attention(nn.Module):
            def __init__(self, dim, heads=8, head_dim=64, dropout=0.0):
        """
        self.attn = nn.Sequential(
            nn.LayerNorm(in_channel),
            Attention(in_channel, heads, head_dim, attn_drop),
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(in_channel),
            Feedforward(in_channel, hidden_dim, ffn_drop)
        )

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                self.attn,
                self.ffn
            ]))


    def forward(self, x):
        for attn, ffn in self.layers:

            x = attn(x) + x
            x = ffn(x) + x

        return x

class MV2Block(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, expansion=4):
        super(MV2Block, self).__init__()
        self.stride = stride
        self.expansion = expansion
        self.use_res_connect = self.stride == 1 and in_channel == out_channel
        assert stride in [1, 2]

        hidden_dim = int(in_channel * expansion)

        if expansion == 1:
            # dw
            self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)
            self.bn1 = nn.BatchNorm2d(hidden_dim)
            self.act1 = nn.SiLU()
            # pw
            self.conv2 = nn.Conv2d(hidden_dim, out_channel, 1, 1, 0, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channel)

        else:
            # pw
            self.conv1 = nn.Conv2d(in_channel, hidden_dim, 1, 1, 0, bias=False)
            self.bn1 = nn.BatchNorm2d(hidden_dim)
            self.act1 = nn.SiLU()

            # dw
            self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)
            self.bn2 = nn.BatchNorm2d(hidden_dim)
            self.act2 = nn.SiLU()

            # pw ->
            self.conv3 = nn.Conv2d(hidden_dim, out_channel, 1, 1, 0, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        res = x
        if self.expansion == 1:
            # pw
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.act1(x)

            # dw
            x = self.conv2(x)
            x = self.bn2(x)
            if self.use_res_connect:
                return x + res
            else:
                return x

        else:
            # pw
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.act1(x)

            # dw
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.act2(x)

            # linear
            x = self.conv3(x)
            x = self.bn3(x)

            #print(f'res shape: {res.shape}')
            #print(f'x shape: {x.shape}')

            if self.use_res_connect:
                #print(f"x + res: {(x + res).shape}")
                return x + res
            else:
                #print(f'x: {x.shape}')
                return x

class MobileVitBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, hidden_dim, attn_drop=0.0, ffn_drop=0.0):
        super(MobileVitBlock, self).__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, hidden_dim, attn_drop)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()

        # local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # global representations
        _, _, h, w = x.shape
        #print(f'x shape : {x.shape}')
        x = einops.rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = einops.rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # fusion
        x = self.conv3(x)
        #print(f'x shape {x.shape}')
        #print(f'y shape {y.shape}')

        x = torch.cat((x, y), dim=1)
        x = self.conv4(x)

        return x

class MobileVit(nn.Module):
    def __init__(self, img_size, dims, channels, num_classes, expansion=4, kernel_size=3, patch_size=(2, 2)):
        super(MobileVit, self).__init__()
        ih, iw = img_size
        ph, pw = patch_size

        assert ih % ph == 0 and iw % pw == 0

        depth = [2, 4, 3]

        self.conv1 = conv_nxn_bn(3, channels[0], stride=2)

        self.mv2 = nn.ModuleList([])
        self.mv2.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.mv2.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.mv2.append(MV2Block(channels[3], channels[4], 2, expansion))
        self.mv2.append(MV2Block(channels[5], channels[6], 2, expansion))
        self.mv2.append(MV2Block(channels[7], channels[8], 2, expansion))

        self.mvit = nn.ModuleList([])

        self.mvit.append(MobileVitBlock(dims[0], depth[0], channels[5], kernel_size, patch_size, int(dims[0] * 2)))
        self.mvit.append(MobileVitBlock(dims[1], depth[1], channels[7], kernel_size, patch_size, int(dims[1] * 4)))
        self.mvit.append(MobileVitBlock(dims[2], depth[2], channels[9], kernel_size, patch_size, int(dims[2] * 4)))

        self.conv2 = conv_1x1_bn(channels[-2], channels[-1])

        self.avg_pool = nn.AvgPool2d(ih // 32, 1)
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.mv2[0](x)

        x = self.mv2[1](x)
        x = self.mv2[2](x)
        x = self.mv2[3](x)

        x = self.mv2[4](x)
        x = self.mvit[0](x)

        x = self.mv2[5](x)
        x = self.mvit[1](x)

        x = self.mv2[6](x)
        x = self.mvit[2](x)
        x = self.conv2(x)

        x = self.avg_pool(x).view(-1, x.shape[1])
        x = self.fc(x)

        return x

def mobilevit_xxs():
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    return MobileVit((512, 512), dims, channels, num_classes=1000, expansion=2)


def mobilevit_xs():
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    return MobileVit((512, 512), dims, channels, num_classes=1000)


def mobilevit_s():
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    return MobileVit((512, 512), dims, channels, num_classes=1000)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    x = torch.randn(1, 3, 512, 512)
    model = mobilevit_xxs()
    y = model(x)
    print(y.shape)
    print(count_parameters(model))

    x = torch.randn(1, 3, 512, 512)
    model = mobilevit_xs()
    y = model(x)
    print(y.shape)
    print(count_parameters(model))

    x = torch.randn(1, 3, 512, 512)
    model = mobilevit_s()
    y = model(x)
    print(y.shape)
    print(count_parameters(model))

