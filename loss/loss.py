import torch
import torch.nn as nn
import torch.nn.functional as F

from loss.MS_SSIM_L1_loss import MS_SSIM_L1_LOSS
from loss.SSIM import SSIM
from utils import distributed as dist


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

    def forward(self, dr, cl):
        mse = F.mse_loss(dr, cl, reduction='mean')
        psnr = 10.0 * torch.log10((self.data_range ** 2) / mse)

        return -psnr


class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()
        self.loss = SSIM()

    def forward(self, dr, cl):
        return -self.loss(dr, cl)


class MS_SSIM_L1_Loss(nn.Module):
    def __init__(self):
        super(MS_SSIM_L1_Loss, self).__init__()
        self.loss = MS_SSIM_L1_LOSS(channel=3, data_range=2., cuda_dev=dist.get_rank())

    def forward(self, dr, cl):
        return self.loss(dr, cl)


class RefGTLoss():
    def __init__(self, args):
        #TODO: create loss get function: calculate + save as object vars
        self.rec_loss = ReconstructionLoss(type=args.rec_loss_type)
        self.ms_ssim_l1_loss = MS_SSIM_L1_Loss() if args.ms_ssim_l1_loss is True else None
        self.ssim_loss = SSIMLoss() if args.ssim_loss is True else None
        self.psnr_loss = PSNRLoss(data_range=2.) if args.psnr_loss is True else None
        
        self.rec_w = 0.5
        self.ssim_w = 0.2
        self.psnr_w = 0.5
        
        self.loss_dict = {}
    
    
    def get_rec_loss(self, dr, cl):
        self.loss_dict['rec_loss'] = self.rec_loss(dr, cl)
        return self.loss_dict['rec_loss']
    
    
    def get_ms_ssim_l1_loss(self, dr, cl):
        self.loss_dict['ms_ssim_l1_loss'] = self.ms_ssim_l1_loss(dr, cl)
        return self.loss_dict['ms_ssim_l1_loss']
    
    
    def get_ssim_loss(self, dr, cl):
        self.loss_dict['ssim_loss'] = self.ssim_loss(dr, cl)
        return self.loss_dict['ssim_loss']
    
    
    def get_psnr_loss(self, dr, cl):
        self.loss_dict['psnr_loss'] = self.psnr_loss(dr, cl)
        return self.loss_dict['psnr_loss']
    
    
    def clear_loss_dict(self):
        self.loss_dict = {}


    def init_loss(self, dr, cl):
        #clear the preview loss dict
        self.clear_loss_dict()
        
        loss = self.get_rec_loss(dr, cl)
        return loss
    
    
    def loss(self, dr, cl):
        #clear the preview loss dict
        self.clear_loss_dict()
        
        loss = self.get_ms_ssim_l1_loss(dr, cl) if (self.ms_ssim_l1_loss) else self.get_rec_loss(dr, cl)
        loss = loss * self.rec_w
        if (self.psnr_loss):
            loss += self.get_psnr_loss(dr, cl) * self.psnr_w
        if (self.ssim_loss):
            loss += self.get_ssim_loss(dr, cl) * self.ssim_w
        return loss


def get_loss_dict(args):
    return RefGTLoss(args)
