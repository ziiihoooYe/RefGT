import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def pool_layer(kernel_size):
    return nn.MaxPool2d(kernel_size=kernel_size, stride=kernel_size)


def up_samp(scale_factor):
    return nn.Upsample(scale_factor=scale_factor, mode='nearest')


### (B, C, N) -> (B, N, C)
def BCN2BNC(x):
    return rearrange(x, 'b c n -> b n c')


### (B, C, N) -> (B, N, C)
def BNC2BCN(x):
    return rearrange(x, 'b n c -> b c n')


### shift = {0, 1, 2, 3}
def patching_layer(x, patch_size, shift):
    patch_stride = int(patch_size / 4)
    padding = {'0': (patch_stride * 0, patch_size - 0 * patch_stride, patch_stride * 0, patch_size - 0 * patch_stride),
               '1': (patch_stride * 1, patch_size - 1 * patch_stride, patch_stride * 1, patch_size - 1 * patch_stride),
               '2': (patch_stride * 2, patch_size - 2 * patch_stride, patch_stride * 2, patch_size - 2 * patch_stride),
               '3': (patch_stride * 3, patch_size - 3 * patch_stride, patch_stride * 3, patch_size - 3 * patch_stride)}

    x = F.pad(x, padding[str(shift)], mode='reflect')  # (B, C0, H+patch_size, W+patch_size)

    x = F.unfold(x, kernel_size=(patch_size, patch_size), stride=patch_size)  # (B, C, N)

    x = BCN2BNC(x)  # (B, N, C)

    return x


def unpatching_layer(x, output_size, patch_size):
    padding = (0, -patch_size, 0, -patch_size)
    output_size = (patch_size + output_size[0], patch_size + output_size[1])

    x = BNC2BCN(x)   # (B, N, C) -> (B, C, N)

    # (B, C, N) -> (B, C0, H+patch_size, W+patch_size)
    x = F.fold(x, output_size=output_size, kernel_size=(patch_size, patch_size), stride=patch_size)

    x = F.pad(x, padding)  # (B, C0, H+patch_size, W+patch_size) -> (B, C0, H, W)

    return x


class Attention(nn.Module):
    def __init__(self, channel_num, heads, heads_feats, dropout=0.):
        super(Attention, self).__init__()
        inner_channel = channel_num * heads
        self.ln = nn.LayerNorm(channel_num)
        self.heads = heads
        self.scale = heads_feats ** -0.5

        self.attn = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(dropout)

        self.q_heads_pro = nn.Linear(in_features=channel_num, out_features=inner_channel)
        self.k_heads_pro = nn.Linear(in_features=channel_num, out_features=inner_channel)
        self.v_heads_pro = nn.Linear(in_features=channel_num, out_features=inner_channel)

        self.to_out = nn.Sequential(
            nn.Linear(inner_channel, channel_num)
        )

    def forward(self, q, k, v):
        ### project to qkv vector q, k, v (B, N, C)
        q = self.ln(q)  # (B, N, C)
        k = self.ln(k)
        v = self.ln(v)

        q = self.q_heads_pro(q)  # (B, N, C) -> (B, N, C*heads)
        k = self.k_heads_pro(k)
        v = self.v_heads_pro(v)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, N, C)*(B, C, N) -> (B, N_q, N_k)
        attn = self.attn(attn)

        x = (attn @ v)  # (B, N_q, N_k)*(B, N, C) -> (B, N, C)

        x = self.to_out(x)  # (B, N, C) -> (B, N, C)

        return x


class FeedForward(nn.Module):
    def __init__(self, channel_num, inner_feats=64, dropout=0.):
        super(FeedForward, self).__init__()
        self.ln = nn.LayerNorm(channel_num)
        self.ff = nn.Sequential(
            nn.Linear(channel_num, inner_feats),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_feats, channel_num)
        )

    def forward(self, x):
        return self.ff(self.ln(x))


class RefTransformer(nn.Module):
    def __init__(self, n_feats, patch_size, heads=3, dropout=0.):
        super(RefTransformer, self).__init__()
        channel_num = n_feats * (patch_size ** 2)

        self.patch_size = patch_size
        self.self_attn = Attention(channel_num=channel_num, heads=heads, heads_feats=channel_num, dropout=dropout)
        self.ref_attn = Attention(channel_num=channel_num, heads=heads, heads_feats=channel_num, dropout=dropout)
        self.ff = FeedForward(channel_num=channel_num, inner_feats=channel_num, dropout=dropout)


    def forward(self, dr_img, dr_ref, cl_ref):
        _, _, h, w = dr_img.size()

        x = patching_layer(dr_img, patch_size=self.patch_size, shift=0)  # (B, N, C)

        ###self attention stage with shift
        ### shift: 0
        # ref_self = patching_layer(dr_img, patch_size=self.patch_size, shift=0).detach()  # self attention
        # x = self.self_attn(q=x, k=ref_self, v=ref_self) + x
        # x = self.ff(x) + x

        # ref attention
        ref_k = patching_layer(dr_ref, patch_size=self.patch_size, shift=0).detach()  # (B, N, C)
        ref_v = patching_layer(cl_ref, patch_size=self.patch_size, shift=0).detach()  # (B, N, C)
        x = self.ref_attn(q=x, k=ref_k, v=ref_v) + x  # (B, N, C)
        x = self.ff(x) + x  # (B, N, C)

        ### shift: 1
        # ref_self = patching_layer(dr_img, patch_size=self.patch_size, shift=1).detach()  # self attention
        # x = self.self_attn(q=x, k=ref_self, v=ref_self) + x
        # x = self.ff(x) + x

        # ref attention
        ref_k = patching_layer(dr_ref, patch_size=self.patch_size, shift=1).detach()
        ref_v = patching_layer(cl_ref, patch_size=self.patch_size, shift=1).detach()
        x = self.ref_attn(q=x, k=ref_k, v=ref_v) + x
        x = self.ff(x) + x

        ### shift: 2
        # ref_self = patching_layer(dr_img, patch_size=self.patch_size, shift=2).detach()  # self attention
        # x = self.self_attn(q=x, k=ref_self, v=ref_self) + x
        # x = self.ff(x) + x

        # ref attention
        ref_k = patching_layer(dr_ref, patch_size=self.patch_size, shift=2).detach()
        ref_v = patching_layer(cl_ref, patch_size=self.patch_size, shift=2).detach()
        x = self.ref_attn(q=x, k=ref_k, v=ref_v) + x
        x = self.ff(x) + x

        ### shift: 3
        # ref_self = patching_layer(dr_img, patch_size=self.patch_size, shift=3).detach()  # self attention
        # x = self.self_attn(q=x, k=ref_self, v=ref_self) + x
        # x = self.ff(x) + x

        # ref attention
        ref_k = patching_layer(dr_ref, patch_size=self.patch_size, shift=3).detach()
        ref_v = patching_layer(cl_ref, patch_size=self.patch_size, shift=3).detach()
        x = self.ref_attn(q=x, k=ref_k, v=ref_v) + x
        x = self.ff(x) + x

        x = unpatching_layer(x, (h, w), patch_size=self.patch_size)  # (B, C, H, W)

        return x
