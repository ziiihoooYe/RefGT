import torch
import torch.nn as nn
import torch.nn.functional as F

from loss.MS_SSIM_L1_loss import MS_SSIM_L1_LOSS
from loss.SSIM import SSIM


class ReconstructionLoss(nn.Module):
    def __init__(self, type='l1'):
        super(ReconstructionLoss, self).__init__()
        if (type == 'l1'):
            self.loss = nn.L1Loss()
        elif (type == 'l2'):
            self.loss = nn.MSELoss()
        else:
            raise SystemExit('Error: no such type of ReconstructionLoss!')

    def forward(self, dr, cl_img):
        return self.loss(dr, cl_img)


class PSNRLoss(nn.Module):
    def __init__(self, data_range):
        super(PSNRLoss, self).__init__()
        self.data_range = data_range

    def forward(self, img1, img2):
        mse = F.mse_loss(img1, img2, reduction='mean')
        psnr = 10.0 * torch.log10((self.data_range ** 2) / mse)

        return -psnr



class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()
        self.loss = SSIM()

    def forward(self, img1, img2):
        return -self.loss(img1, img2)


class MS_SSIM_L1_Loss(nn.Module):
    def __init__(self):
        super(MS_SSIM_L1_Loss, self).__init__()
        self.loss = MS_SSIM_L1_LOSS(channel=3, data_range=2.)

    def forward(self, img1, img2):
        return self.loss(img1, img2)


def get_loss_dict(args):
    loss = {}
    loss['ms_ssim_l1_loss'] = MS_SSIM_L1_Loss()
    loss['rec_loss'] = ReconstructionLoss(type=args.rec_loss_type)
    if (args.psnr_loss):
        loss['psnr_loss'] = PSNRLoss(data_range=2.)
    if (args.ssim_loss):
        loss['psnr_loss'] = SSIMLoss()

    return loss