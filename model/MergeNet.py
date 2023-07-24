import torch
import torch.nn as nn
import torch.nn.functional as F

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=True)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)

    def forward(self, x):
        x1 = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.res_scale + x1
        return out


### shallow feature extractor
class SFE(nn.Module):
    def __init__(self, num_res_blocks, n_feats, res_scale):
        super(SFE, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.conv_head = conv3x3(n_feats, n_feats)

        self.RBs = nn.ModuleList()
        for i in range(self.num_res_blocks):
            self.RBs.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                                     res_scale=res_scale))

        self.conv_tail = conv3x3(n_feats, n_feats)

    def forward(self, x):
        x = F.relu(self.conv_head(x))
        x1 = x
        for i in range(self.num_res_blocks):
            x = self.RBs[i](x)
        x = self.conv_tail(x)
        x = x + x1
        return x


class TextureFusion(nn.Module):
    def __init__(self, in_channels, out_channels, num_res_blocks, res_scale):
        super(TextureFusion, self).__init__()
        self.conv_head = conv3x3(in_channels, out_channels)
        self.conv = nn.Sequential()
        for i in range(num_res_blocks):
            self.conv.append(ResBlock(in_channels=out_channels, out_channels=out_channels, res_scale=res_scale))

    def forward(self, x, S, T):
        x_res = x

        x = torch.cat((x, T), dim=1)
        x = self.conv_head(x)
        x = self.conv(x)
        x = x * S + x_res

        return x


class MergeNet(nn.Module):
    def __init__(self, num_res_blocks, n_feats, res_scale):
        super(MergeNet, self).__init__()
        self.num_res_blocks = num_res_blocks ### a list containing number of resblocks of different stages
        self.n_feats = n_feats

        self.SFE = SFE(self.num_res_blocks[0], n_feats, res_scale)

        self.TF0 = TextureFusion(in_channels=n_feats*2, out_channels=n_feats, num_res_blocks=num_res_blocks[1],
                                 res_scale=res_scale)
        self.TF1 = TextureFusion(in_channels=n_feats*2, out_channels=n_feats, num_res_blocks=num_res_blocks[2],
                                 res_scale=res_scale)
        self.TF2 = TextureFusion(in_channels=n_feats*2, out_channels=n_feats, num_res_blocks=num_res_blocks[3],
                                 res_scale=res_scale)

    def forward(self, x, S, T):
        x = self.SFE(x)

        x = self.TF0(x, S=S, T=T)

        x = self.TF1(x, S=S, T=T)

        x = self.TF2(x, S=S, T=T)

        return x
