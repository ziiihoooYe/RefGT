import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)


class UpsamHead(nn.Module):
    def __init__(self, n_feats):
        super(UpsamHead, self).__init__()
        self.conv1 = conv3x3(n_feats, n_feats * 2)
        self.conv2 = conv3x3(n_feats * 2, n_feats * 2)
        self.conv3 = conv3x3(n_feats * 2, n_feats * 4)
        self.conv4 = conv3x3(n_feats * 4, n_feats * 4)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = F.relu(self.conv4(x))

        return x


class ResBlock(nn.Module):
    def __init__(self, in_n_feats, out_n_feats, stride=1, res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = conv3x3(in_n_feats, out_n_feats, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_n_feats, out_n_feats)

    def forward(self, x):
        x1 = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.res_scale + x1
        return out


class MergeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, res_scale=1):
        super(MergeBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, T):
        x_res = torch.cat((x, T), 1)
        x_res = self.conv1(x_res)
        x_res = self.relu(x_res)
        out = x_res * self.res_scale + x
        return out


class NetTail(nn.Module):
    def __init__(self, n_feats, res_depth):
        super(NetTail, self).__init__()
        self.res_depth = res_depth

        self.n_feats = n_feats

        self.merge_block = conv3x3(n_feats * 2, n_feats)

        self.res_block = ResBlock(n_feats, n_feats)

        self.merge_tail = conv3x3(n_feats, 3)

        self.tail = ResBlock(3, 3)

    def forward(self, x, dr_img_T, dr_img):
        for i in range(self.res_depth):
            x = torch.cat((x, dr_img_T), dim=1)
            x = F.relu(self.merge_block(x))
            x = self.res_block(x)

        x = F.relu(self.merge_tail(x))
        x = self.tail(x) + dr_img

        return x


### lv2: bigger scale, lv1: smaller scale, in_feats: channel number of lv1
class TextureFusion(nn.Module):
    def __init__(self, in_feats, out_feats, res_depth):
        super(TextureFusion, self).__init__()
        self.res_depth = res_depth

        self.lv2_head = conv3x3(in_feats*2, in_feats*4)
        self.lv1_head = conv3x3(in_feats, in_feats)

        self.up_sam = nn.PixelShuffle(upscale_factor=2)

        self.merge_block = conv3x3(in_feats*2, in_feats)
        self.res_block = ResBlock(in_feats, out_feats)


    def forward(self, T_lv1, T_lv2):
        T_lv2 = F.relu(self.lv2_head(T_lv2))
        T_lv2 = self.up_sam(T_lv2)

        for i in range(self.res_depth):
            x = torch.cat((T_lv1, T_lv2), dim=1)
            x = F.relu(self.merge_block(x))
            x = self.res_block(x)

        return x

