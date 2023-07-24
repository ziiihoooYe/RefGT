import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt

from model import LTE, MergeNet, TextureFusion, SearchTransfer, RefTransformer, MainNet


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)


def fea_map(feature, scale):
    fea_maps = []
    num = feature.size(0)
    feature = np.transpose(feature.cpu().numpy(), (1, 2, 0))
    feature = feature.repeat(scale, axis=0).repeat(scale, axis=1)
    for i in range(num):
        fea_maps.append(feature[:,:,i])
    _fea_map = sum(e for e in fea_maps)
    plt.imshow(_fea_map)
    return _fea_map

def show_img(img):
    img = (img+1.)/2
    img = np.transpose(img.cpu().numpy(), (1, 2, 0))
    plt.imshow(img)
    return img

class DRTT(nn.Module):
    def __init__(self, args):
        super(DRTT, self).__init__()
        self.args = args
        res_depth = list(map(int, args.res_depth.split('+')))

        self.LTE = LTE.LTE(requires_grad=args.lte_grad, mean_grad=args.mean_grad)
        self.LTE_copy = LTE.LTE(requires_grad=False)
        self.SearchTransfer = SearchTransfer.SearchTransfer()
        self.MainNet = MainNet.MainNet(num_res_blocks=res_depth, n_feats=args.n_feats, res_scale=args.res_scale)

    def forward(self, dr_img=None, cl_ref=None, dr_ref=None):
        # if (type(gt) != type(None)):
        #     ### used in transferal perceptual loss
        #     self.LTE_copy.load_state_dict(self.LTE.state_dict())
        #     gt_lv1, gt_lv2, gt_lv3 = self.LTE_copy((gt + 1.) / 2.)
        #     return gt_lv3, gt_lv2, gt_lv1
        #
        # if LTE_auto_encoder:
        #     lv1, lv2, lv3, x_lv1, x_lv2, x_lv3 = self.LTE((dr_img.detach() + 1.) / 2, LTE_auto_encoder)
        #     return lv1, lv2, lv3, x_lv1, x_lv2, x_lv3

        _, _, dr_img_T_lv3 = self.LTE((dr_img.detach() + 1.) / 2)
        _, _, dr_ref_T_lv3 = self.LTE((dr_ref.detach() + 1.) / 2)
        cl_ref_T_lv1, cl_ref_T_lv2, cl_ref_T_lv3 = self.LTE((cl_ref.detach() + 1.) / 2)

        S, T_lv3, T_lv2, T_lv1 = self.SearchTransfer(dr_img_T_lv3, dr_ref_T_lv3,
                                                     cl_ref_T_lv1, cl_ref_T_lv2, cl_ref_T_lv3, cl_ref, dr_img)

        dr = self.MainNet(dr_img, S, T_lv3, T_lv2, T_lv1)

        return dr, S, T_lv3, T_lv2, T_lv1
