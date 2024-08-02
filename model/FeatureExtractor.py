import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


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


class LTE(torch.nn.Module):
    def __init__(self, requires_grad=True, mean_grad=False, rgb_range=1):
        super(LTE, self).__init__()

        # use vgg19 weights to initialize
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
        self.sub_mean = MeanShift(
            mean_grad=mean_grad, rgb_range=rgb_range, rgb_mean=vgg_mean, rgb_std=vgg_std)

    def forward(self, x):
        x = self.sub_mean(x)
        x_lv1 = self.slice1(x)  # (B, 64, H, W)

        x_lv2 = self.slice2(x_lv1)  # (B, 128, H/2, W/2)

        x_lv3 = self.slice3(x_lv2)  # (B, 258, H/4, W/4)

        return x_lv1, x_lv2, x_lv3
