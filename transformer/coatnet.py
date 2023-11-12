import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange

class conv_3x3_bn(nn.Module):
    def __init__(self, input, output, img_size, downsample=False):
        super(conv_3x3_bn, self).__init__()
        self.stride = 1 if downsample == False else 2
        self.conv2d = nn.Conv2d(input, output, 3, stride=self.stride, bias=False)
        self.bn2d = nn.BatchNorm2d(output)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn2d(x)
        x = self.gelu(x)

        return x


class pre_norm(nn.Module):
    def __init__(self, dim, fn, norm):
        super(pre_norm, self).__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)




class SEModule(nn.Module):
    '''
    SEmodule, 用于模型中的特征重标定，首先对特征图进行全局平均池化以获得每个
    通道的全局信息，然后通过两个全连接层来获得每个通道的权重，最后对原始特侦图
    进行通道注意力调制

    SE模组插入某些块中，增强模型对通道之间关系的捕获能力
    '''
    def __init__(self, in_channel, out_channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(out_channel, in_channel // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channel // reduction, out_channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x

class DWConv(nn.Module):
    '''
    DW，减少计算量和参数量，
    第一步是对每一个输入通道进行卷积操作，也就是深度卷积
    第二布常用1x1卷积，点卷积，来组合前一步的输出
    '''
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0):
        super(DWConv, self).__init__()
        # dwconv
        self.depthwise = nn.Conv2d(in_channel, in_channel, kernel_size, stride, padding, groups=in_channel, bias=False)
        # pwconv
        self.pointwise = nn.Conv2d(in_channel, out_channel, 1, 1, 0, 1, bias=False)

    def forward(self, x):
        x = self.pointwise(self.depthwise(x))

        return x




class MBConv(nn.Module):
    def __init__(self, in_channel, out_channel, img_size, downsample=False, expansion=4):
        super(MBConv, self).__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2

        hidden_dim = int(in_channel * expansion)

        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False)

        if expansion == 1:
            self.conv = nn.Sequential(
                #dw
                nn.Conv2d(in_channel, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),

                #pw
                nn.Conv2d(hidden_dim, out_channel, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channel)
            )
        else:
            self.conv = nn.Sequential(
                #pw down_sample
                nn.Conv2d(in_channel, hidden_dim, 1, stride, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),

                #dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                SEModule(in_channel, hidden_dim),

                # linear
                nn.Conv2d(hidden_dim, out_channel, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channel)
            )

        self.pre_conv = pre_norm(in_channel, self.conv, nn.BatchNorm2d)
        self.norm = nn.BatchNorm2d(in_channel)

    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(self.norm(x))

        else:
            return x + self.conv(self.norm(x))



class Attention(nn.Module):
    def __init__(self, in_channel, out_channel, img_size, heads=8, head_channel=32, dropout=0.0):
        super(Attention, self).__init__()
        inner_channel = head_channel * heads
        self.img_h, self.img_w = img_size

        project_out = not (heads == 1 and head_channel == inner_channel)

        self.heads = heads
        self.scale = head_channel ** -0.5

        # Relative position bias
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.img_h - 1) * (2 * self.img_w - 1), heads)
        )

        coords = torch.meshgrid((torch.arange(self.img_h), torch.arange(self.img_w)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]

        relative_coords[0] += self.img_h - 1
        relative_coords[1] += self.img_w - 1
        relative_coords[0] *= 2 * self.img_w - 1
        relative_coords = relative_coords.permute(1, 2, 0)
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.attn = nn.Softmax(dim=-1)
        self.to_qvk = nn.Linear(in_channel, inner_channel * 3, bias=False)

        self.out = nn.Sequential(
            nn.Linear(inner_channel, out_channel),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()


    def forward(self, x):
        qkv = self.to_qvk(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h=self.heads) , qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # gather
        relative_bias = self.relative_bias_table.gather(
            0, self.relative_index.repeat(1, self.heads)
        )
        relative_bias = einops.rearrange(relative_bias, '(h w) c -> 1 c h w', h=self.img_h * self.img_w, w=self.img_h * self.img_w)

        dots = dots + relative_bias

        attn = self.attn(dots)

        out = torch.matmul(attn, v)
        out = einops.rearrange(out, 'b h n d -> b n (h d)')
        out = self.out(out)

        return out




class Transformer(nn.Module):
    def __init__(self, in_channel, out_channel, img_size, heads=8, head_channel=32, downsample=True, dropout=0.0):
        super(Transformer, self).__init__()

        self.img_w, self.img_h = img_size
        self.hidden_dim = int(in_channel * 4)
        self.downsample = downsample

        if self.downsample:
            self.pool1 = nn.MaxPool2d(3, 2, 1)
            self.pool2 = nn.MaxPool2d(3, 2, 1)
            self.proj  = nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False)

        self.attn = Attention(in_channel, out_channel, img_size, heads, head_channel, dropout)
        self.feed_forward = FeedForward(out_channel, self.hidden_dim, dropout)

        self.attn = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            nn.LayerNorm(in_channel),
            self.attn,
            Rearrange('b (h w) c -> b c h w', h=self.img_h, w=self.img_w)
        )

        self.feed_forward = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            nn.LayerNorm(out_channel),
            self.feed_forward,
            Rearrange('b (h w) c -> b c h w', h = self.img_h, w=self.img_w)
        )

    def forward(self, x):
        if self.downsample:
            x = self.proj(self.pool1(x)) + self.attn(self.pool2(x))
        else:
            x = x + self.attn(x)

        x = x + self.feed_forward(x)

        return x


class CoAtNet(nn.Module):
    def __init__(self, img_size, in_channel, num_blocks, channel, num_classes,):
        block_type = ['C', 'C', 'T', 'T']
        super(CoAtNet, self).__init__()

        self.img_h, self.img_w = img_size
        blocks = {'C': MBConv, 'T': Transformer}

        self.s_0 = self.make_layer_(
            img_size=(self.img_h // 2, self.img_w // 2),
            blocks=conv_3x3_bn,
            in_channel=in_channel,
            out_channel=channel[0],
            depth=num_blocks[0]
        )

        self.s_1 = self.make_layer_(
            img_size=(self.img_h // 4, self.img_h // 4),
            blocks=blocks['C'],
            in_channel=channel[0],
            out_channel=channel[1],
            depth=num_blocks[1]
        )

        self.s_2 = self.make_layer_(
            img_size=(self.img_h // 8, self.img_w // 8),
            blocks=blocks['C'],
            in_channel=channel[1],
            out_channel=channel[2],
            depth=num_blocks[2]
        )

        self.s_3 = self.make_layer_(
            img_size=(self.img_h // 16, self.img_w // 16),
            blocks=blocks['T'],
            in_channel=channel[2],
            out_channel=channel[3],
            depth=num_blocks[3]
        )

        self.s_4 = self.make_layer_(
            img_size=(self.img_h // 32, self.img_w // 32),
            blocks=blocks['T'],
            in_channel=channel[3],
            out_channel=channel[4],
            depth=num_blocks[4]
        )

        self.avg_pool = nn.AvgPool2d(self.img_h // 32, 1)
        self.fc = nn.Linear(channel[-1], num_classes, bias=False)

    def forward(self, x):
        #x = self.s_4(self.s_3(self.s_2(self.s_1(self.s_0(x)))))
        #x = self.avg_pool(x).view(-1, x.shape[1])
        #x = self.fc(x)

        x_0 = self.s_0(x)
        print(f'x_0: {x_0.shape}')

        x_1 = self.s_1(x_0)
        print(f'x_1: {x_1.shape}')

        x_2 = self.s_2(x_1)
        print(f'x_2: {x_2.shape}')

        x_3 = self.s_3(x_2)
        print(f'x_3: {x_3.shape}')

        x_4 = self.s_4(x_3)
        print(f'x_4: {x_4.shape}')

        x_5 = self.avg_pool(x_4).view(-1, x_4.shape[1])
        print(f'x_5: {x_5.shape}')

        x_6 = self.fc(x_5)
        print(f'x_6: {x_6.shape}')

        return x


    def make_layer_(self, img_size, blocks, in_channel, out_channel, depth):

        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(blocks(in_channel, out_channel, img_size, downsample=True))
            else:
                layers.append(blocks(out_channel, out_channel, img_size))
        return nn.Sequential(*layers)


'''
    def __init__(self, img_size, in_channel, num_blocks, channel, num_classes,):
'''

def coatnet_0(img_size, num_classes=5):
    num_blocks = [2, 2, 3, 5, 2]            # L
    channels = [64, 96, 192, 384, 768]      # D
    return CoAtNet(img_size, 3, num_blocks, channels, num_classes=num_classes)


def coatnet_1(img_size, num_classes=5):
    num_blocks = [2, 2, 6, 14, 2]           # L
    channels = [64, 96, 192, 384, 768]      # D
    return CoAtNet(img_size, 3, num_blocks, channels, num_classes=num_classes)


def coatnet_2(img_size, num_classes=5):
    num_blocks = [2, 2, 6, 14, 2]           # L
    channels = [128, 128, 256, 512, 1026]   # D
    return CoAtNet(img_size, 3, num_blocks, channels, num_classes=num_classes)


def coatnet_3(img_size, num_classes=5):
    num_blocks = [2, 2, 6, 14, 2]           # L
    channels = [192, 192, 384, 768, 1536]   # D
    return CoAtNet(img_size, 3, num_blocks, channels, num_classes=num_classes)


def coatnet_4(img_size, num_classes=5):
    num_blocks = [2, 2, 12, 28, 2]          # L
    channels = [192, 192, 384, 768, 1536]   # D
    return CoAtNet(img_size, 3, num_blocks, channels, num_classes=num_classes)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    img = torch.randn(1, 3, 224, 224)
    print(img.size())
    model = coatnet_0((224, 224), 5)

    out = model(img)

    #print(out.shape, count_parameters(model))
