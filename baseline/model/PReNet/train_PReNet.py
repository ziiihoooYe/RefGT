import os
import sys
# sys.path.append(os.getcwd())
# print(sys.path)
# print(os.getcwd())
sys.path.insert(0, os.getcwd())
sys.path.insert(1, os.path.join(os.getcwd(), 'baseline/model/PReNet'))
# os.chdir('baseline/model/PReNet')
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
# from DerainDataset i#mport *
from torch.optim.lr_scheduler import MultiStepLR
from loss.SSIM import SSIM
from networks import *
from option import parser
from dataloader import get_dataloader
from utils.utils import *
from utils.matrics import matrics_update
from baseline_utils import findLastCheckpoint
# os.chdir('../../..')

parser.add_argument("--preprocess", type=bool, default=True, help='run prepare_data or not')
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=[6, 10, 16], help="When to decay learning rate")
parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
parser.add_argument("--save_path", type=str, default="baseline/state_dict/PReNet6", help='path to save models and log files')
parser.add_argument("--save_freq",type=int,default=5, help='save intermediate model')
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')
parser.add_argument('--num_gpu', type=int, default=8)
parser.add_argument('--resume', type=bool, default=True)
args = parser.parse_args()
args.patch_size = 8
# args.dataset_dir = os.path.join('../../..', args.dataset_dir)

if args.use_gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"


def narrow_img(img, img_size):
    output = []
    for i in range(img.size(0)):
        img_h = img[i].size(-2)  # (C, H, W)
        img_start_top = img_h - img_size[i][-2]  # (H, W)
        img_w = img[i].size(-1)
        img_start_left = img_w - img_size[i][-1]
        _img = img[i].narrow(1, img_start_top, img_h - img_start_top).narrow(2, img_start_left, img_w - img_start_left)
        output.append(_img)
    return output


def tensor2img(tensor):
    img = tensor * 255.
    img = img.detach().squeeze().round().cpu()

    return img


def main():
    print('Loading dataset ...\n')
    ### dataloader of training set and testing set
    dataloader = get_dataloader(args)
    print("# of training samples: %d\n" % int(len(dataloader['train'])))

    # Build model
    model = PReNet(recurrent_iter=args.recurrent_iter, use_GPU=args.use_gpu)
    # print_network(model)

    # loss function
    criterion = SSIM()

    # Move to GPU
    if args.use_gpu:
        if args.num_gpu > 1:
            model = nn.DataParallel(model, list(range(args.num_gpu)))
        model = model.cuda()
        criterion = criterion.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=args.milestone, gamma=0.2)  # learning rates

    # record training
    # writer = SummaryWriter(args.save_path)

    # load the lastest model
    if args.resume:
        initial_epoch = findLastCheckpoint(save_dir=os.path.join(args.save_path, 'Cityscapes'))
    else:
        initial_epoch = 0
        
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        model.load_state_dict(torch.load(os.path.join(args.save_path, args.dataset, 'net_epoch%d.pth' % initial_epoch)))

    # start training
    step = 0
    psnr_train = 0.
    ssim_train = 0.
    for epoch in range(initial_epoch, args.epochs):
        scheduler.step(epoch)
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])

        ## epoch training start
        for i, sampled_batch in enumerate(dataloader['train']):
            torch.cuda.empty_cache()

            input_train = sampled_batch['rn_img']  # (-1, 1)
            input_train = (input_train + 1.) / 2  # (-1, 1) -> (0, 1)
            target_train = sampled_batch['cl_img']  # (-1, 1)
            target_train = (target_train + 1.) / 2  # (-1, 1) -> (0, 1)
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            # input_train, target_train = Variable(input_train), Variable(target_train)

            if args.use_gpu:
                input_train, target_train = input_train.cuda(), target_train.cuda()

            out_train, _ = model(input_train)
            pixel_metric = criterion(target_train, out_train)
            loss = -pixel_metric

            loss.backward()
            optimizer.step()

            # training curve
            model.eval()
            out_train, _ = model(input_train)
            out_train = torch.clamp(out_train, 0., 1.)
            psnr_train, ssim_train = matrics_update(psnr_train, ssim_train, i+1, out_train.detach(), target_train.detach())
            print("[epoch %d][%d/%d] loss: %.4f, pixel_metric: %.4f, PSNR: %.4f" %
                  (epoch+1, i+1, len(dataloader['train']), loss.item(), pixel_metric.item(), psnr_train))

        save_path = os.path.join(args.save_path, args.dataset)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # save model
        torch.save(model.state_dict(), os.path.join(save_path, 'net_latest.pth'))
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(save_path, 'net_epoch%d.pth' % (epoch+1)))


if __name__ == "__main__":
    main()
