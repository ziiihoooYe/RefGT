import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# def conv3x3(in_channels, out_channels, stride=1):
#     return nn.Conv2d(in_channels, out_channels, kernel_size=3,
#                      stride=stride, padding=1, bias=True)


class MeanShift(nn.Conv2d):
    def __init__(self, mean_grad=False, rgb_range=1, rgb_mean=(0.485, 0.456, 0.406),
                 rgb_std=(0.229, 0.224, 0.225), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=(1, 1))
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)

        if not mean_grad:
            self.weight.requires_grad = mean_grad
            self.bias.requires_grad = mean_grad


#
# class ResBlock(nn.Module):
#     def __init__(self, in_n_feats, out_n_feats, stride=1, res_scale=1):
#         super(ResBlock, self).__init__()
#         self.res_scale = res_scale
#         self.conv1 = conv3x3(in_n_feats, out_n_feats, stride)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(out_n_feats, out_n_feats)
#
#     def forward(self, x):
#         x1 = x
#         out = self.conv1(x)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = out * self.res_scale + x1
#         return out


class LTE(torch.nn.Module):
    def __init__(self, requires_grad=True, mean_grad=False, rgb_range=1):
        super(LTE, self).__init__()

        ### use vgg19 weights to initialize
        vgg_pretrained_features = models.vgg19(pretrained=True).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.slice1.parameters():
                param.requires_grad = requires_grad
            for param in self.slice2.parameters():
                param.requires_grad = requires_grad
            for param in self.slice3.parameters():
                param.requires_grad = requires_grad

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = MeanShift(mean_grad=mean_grad, rgb_range=rgb_range, rgb_mean=vgg_mean, rgb_std=vgg_std)

    def forward(self, x):
        x = self.sub_mean(x)
        x_lv1 = self.slice1(x)  # (B, 64, H, W)

        x_lv2 = self.slice2(x_lv1)  # (B, 128, H/2, W/2)

        x_lv3 = self.slice3(x_lv2)  # (B, 258, H/4, W/4)

        return x_lv1, x_lv2, x_lv3

# class channel_projector(nn.Module):
#     def __init__(self, in_feats, out_feats):
#         super(channel_projector, self).__init__()
#
#         self.head = conv3x3(in_feats, out_feats)
#         self.tail = ResBlock(out_feats, out_feats)
#
#     def forward(self, x):
#         x = F.relu(self.head(x))
#         x = self.tail(x)
#
#         return x
