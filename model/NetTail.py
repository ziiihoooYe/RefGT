import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)

class UpsamHead(nn.Module):
    def __init__(self, n_feats):
        super(UpsamHead, self).__init__()
        self.conv1 = conv3x3(n_feats, n_feats*2)
        self.conv2 = conv3x3(n_feats*2, n_feats*2)
        self.conv3 = conv3x3(n_feats*2, n_feats*4)
        self.conv4 = conv3x3(n_feats*4, n_feats*4)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = F.relu(self.conv4(x))

        return x


class NetTail(nn.Module):
    def __init__(self, n_feats):
        super(NetTail, self).__init__()

        self.n_feats = n_feats

        self.tail = nn.Sequential(
            conv3x3(n_feats, n_feats),
            conv3x3(n_feats, n_feats),
            conv3x3(n_feats, n_feats),
            conv3x3(n_feats, n_feats),
            nn.ReLU(),
            conv3x3(n_feats, 3)
        )
        # self.tail.append(conv3x3(n_feats+3, n_feats))
        # self.tail.append(nn.ReLU())
        # self.tail.append(conv3x3(n_feats, n_feats))
        # self.tail.append(conv3x3(n_feats, 3))

    def forward(self, x, dr_img):

        # x = torch.cat((x, dr_img), dim=1)
        x = self.tail(x)

        # x = x + dr_img

        return x
