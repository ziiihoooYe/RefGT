import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import utils.distributed as dist


class SearchTransfer(nn.Module):
    def __init__(self, save_dir):
        super(SearchTransfer, self).__init__()

    def forward(self, query_lv3, key_lv3, value_lv1, value_lv2, value_lv3, cl_ref, dr_img):
        k = 1
        n = 10
        kernel_size = 3
        padding = 1

        with torch.no_grad():

            # search
            query_unfold = F.unfold(query_lv3, kernel_size=(kernel_size, kernel_size), padding=padding)
            key_unfold = F.unfold(key_lv3, kernel_size=(kernel_size, kernel_size), padding=padding)
            key_unfold = key_unfold.permute(0, 2, 1)

            key_unfold = F.normalize(key_unfold, dim=2)  # [N, Hr*Wr, C*k*k]
            query_unfold = F.normalize(query_unfold, dim=1)  # [N, C*k*k, H*W]

            relevance = torch.bmm(key_unfold, query_unfold)  # [N, Hr*Wr, H*W]
            relevance, relevance_idx = torch.topk(relevance, k=k, dim=1)  # [N, k, H*W]

            # transfer
            value_lv3_unfold = F.unfold(value_lv3, kernel_size=(kernel_size, kernel_size), padding=padding)  # [N, C3, Hr*Wr]
            value_lv2_unfold = F.unfold(value_lv2, kernel_size=(2*kernel_size, 2*kernel_size), padding=2*padding, stride=2)  # [N, C2, Hr*Wr]
            value_lv1_unfold = F.unfold(value_lv1, kernel_size=(4*kernel_size, 4*kernel_size), padding=4*padding, stride=4)  # [N, C1, Hr*Wr]

            # expand idx
            relevance_lv3_idx = relevance_idx.unsqueeze(1).expand(-1, value_lv3_unfold.size(1), -1, -1).transpose(-2, -1)  # [N, C3, H*W, k]
            relevance_lv2_idx = relevance_idx.unsqueeze(1).expand(-1, value_lv2_unfold.size(1), -1, -1).transpose(-2, -1)  # [N, C2, H*W, k]
            relevance_lv1_idx = relevance_idx.unsqueeze(1).expand(-1, value_lv1_unfold.size(1), -1, -1).transpose(-2, -1)  # [N, C1, H*W, k]
            relevance = relevance.unsqueeze(1).transpose(-2, -1)  # [N, 1, H*W, k]

            # gather value
            topk_value_lv3 = value_lv3_unfold.unsqueeze(-2).expand(-1, -1, query_unfold.size(-1), -1).gather(-1, relevance_lv3_idx)  # [N, C3, H*W, k]
            topk_value_lv2 = value_lv2_unfold.unsqueeze(-2).expand(-1, -1, query_unfold.size(-1), -1).gather(-1, relevance_lv2_idx)  # [N, C2, H*W, k]
            topk_value_lv1 = value_lv1_unfold.unsqueeze(-2).expand(-1, -1, query_unfold.size(-1), -1).gather(-1, relevance_lv1_idx)  # [N, C1, H*W, k]
            T_lv3_unfold = torch.sum(relevance * topk_value_lv3, dim=-1) / torch.sum(relevance, dim=-1)  # [N, 1, H*W, k] * [N, C3, H*W, k] -> [N, C3, H*W]
            T_lv2_unfold = torch.sum(relevance * topk_value_lv2, dim=-1) / torch.sum(relevance, dim=-1)
            T_lv1_unfold = torch.sum(relevance * topk_value_lv1, dim=-1) / torch.sum(relevance, dim=-1)

            overlap_cnt_lv3 = F.fold(torch.ones_like(T_lv3_unfold), output_size=query_lv3.size()[-2:],
                                     kernel_size=(kernel_size, kernel_size), padding=padding)
            overlap_cnt_lv2 = F.fold(torch.ones_like(T_lv2_unfold), output_size=(query_lv3.size(2)*2, query_lv3.size(3)*2),
                                     kernel_size=(2*kernel_size, 2*kernel_size), padding=2*padding, stride=2)
            overlap_cnt_lv1 = F.fold(torch.ones_like(T_lv1_unfold), output_size=(query_lv3.size(2)*4, query_lv3.size(3)*4),
                                     kernel_size=(4*kernel_size, 4*kernel_size), padding=4*padding, stride=4)

            T_lv3 = F.fold(T_lv3_unfold, output_size=query_lv3.size()[-2:],
                           kernel_size=(kernel_size, kernel_size), padding=padding) / overlap_cnt_lv3
            T_lv2 = F.fold(T_lv2_unfold, output_size=(query_lv3.size(2)*2, query_lv3.size(3)*2),
                           kernel_size=(2*kernel_size, 2*kernel_size), padding=2*padding, stride=2) / overlap_cnt_lv2
            T_lv1 = F.fold(T_lv1_unfold, output_size=(query_lv3.size(2)*4, query_lv3.size(3)*4),
                           kernel_size=(4*kernel_size, 4*kernel_size), padding=4*padding, stride=4) / overlap_cnt_lv1

            relevance, _ = torch.max(relevance, dim=-1)

            S = relevance.view(relevance.size(0), 1, query_lv3.size(2), query_lv3.size(3))

        return S, T_lv3, T_lv2, T_lv1
