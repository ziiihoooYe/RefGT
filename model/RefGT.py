import os
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import FeatureAttention, FeatureExtractor, FeatureFusion
import utils.distributed as dist


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)


class RefGT(nn.Module):
    def __init__(self, args):
        super(RefGT, self).__init__()
        self.args = args
        res_depth = list(map(int, args.res_depth.split('+')))

        # FE: Feature Extractor Module
        self.FE = FeatureExtractor.LTE(requires_grad=args.lte_grad, mean_grad=args.mean_grad)
        # FA: Feature Attention Module
        self.FA = FeatureAttention.SearchTransfer(save_dir=args.save_dir)
        # FF: Feature Fusion Module
        self.FF = FeatureFusion.MainNet(num_res_blocks=res_depth, n_feats=args.n_feats, res_scale=args.res_scale)


    def forward(self, dr_img=None, cl_ref=None, dr_ref=None):
        # Feature Extractor Stage
        _, _, dr_img_F_lv3 = self.FE((dr_img.detach() + 1.) / 2)
        _, _, dr_ref_F_lv3 = self.FE((dr_ref.detach() + 1.) / 2)
        cl_ref_F_lv1, cl_ref_F_lv2, cl_ref_F_lv3 = self.FE((cl_ref.detach() + 1.) / 2)

        # Feature Attention Stage
        S, F_lv3, F_lv2, F_lv1 = self.FA(dr_img_F_lv3, dr_ref_F_lv3, cl_ref_F_lv1, cl_ref_F_lv2, cl_ref_F_lv3, cl_ref, dr_img)

        # Feature Fusion Stage
        dr = self.FF(dr_img, S, F_lv3, F_lv2, F_lv1)
        
        return dr, S, F_lv3, F_lv2, F_lv1
