import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'baseline/model/PReNet'))
os.chdir('baseline/model/PReNet')
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
# from DerainDataset i#mport *
from utils.utils import *
from torch.optim.lr_scheduler import MultiStepLR
from loss.SSIM import SSIM
from networks import *
from option import parser
from dataloader import get_dataloader
# os.chdir('../../..')

parser.add_argument("--preprocess", type=bool, default=True, help='run prepare_data or not')
parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=[6, 10, 16], help="When to decay learning rate")
parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
parser.add_argument("--save_path", type=str, default="../../state_dict/PReNet6/BDD100K", help='path to save models and log files')
parser.add_argument("--save_freq",type=int,default=5, help='save intermediate model')
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')
args = parser.parse_args()
args.patch_size = 8
args.dataset_dir = os.path.join('../../..', args.dataset_dir)

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
        if (not args.cpu) and (args.num_gpu > 1):
            model = nn.DataParallel(model, list(range(args.num_gpu)))
        model = model.cuda()
        criterion = criterion.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=args.milestone, gamma=0.2)  # learning rates

    # record training
    # writer = SummaryWriter(args.save_path)

    # load the lastest model
    initial_epoch = findLastCheckpoint(save_dir=args.save_path)
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        model.load_state_dict(torch.load(os.path.join(args.save_path, 'net_epoch%d.pth' % initial_epoch)))

    # start training
    step = 0
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
            img_sizes = sampled_batch['rn_img_sizes']
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
            psnr_train = batch_PSNR(out_train, target_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f, pixel_metric: %.4f, PSNR: %.4f" %
                  (epoch+1, i+1, len(dataloader['train']), loss.item(), pixel_metric.item(), psnr_train))

            # if step % 10 == 0:
            #     # Log the scalar values
            #     writer.add_scalar('loss', loss.item(), step)
            #     writer.add_scalar('PSNR on training data', psnr_train, step)
            # step += 1
        ## epoch training end

        # log the images
        # model.eval()
        # out_train, _ = model(input_train)
        # out_train = torch.clamp(out_train, 0., 1.)
        # out_train = narrow_img(out_train, img_sizes)
        # input_train = narrow_img(input_train, img_sizes)
        # target_train = narrow_img(target_train, img_sizes)
        # im_target = utils.make_grid(target_train[0], nrow=8, normalize=True, scale_each=True)
        # im_input = utils.make_grid(input_train[0], nrow=8, normalize=True, scale_each=True)
        # im_derain = utils.make_grid(out_train[0], nrow=8, normalize=True, scale_each=True)
        # im_target = tensor2img(target_train[0])
        # im_input = tensor2img(input_train[0])
        # im_derain = tensor2img(out_train[0])
        # writer.add_image('clean image', im_target, epoch+1)
        # writer.add_image('rainy image', im_input, epoch+1)
        # writer.add_image('deraining image', im_derain, epoch+1)

        # save model
        torch.save(model.state_dict(), os.path.join(args.save_path, 'net_latest.pth'))
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(args.save_path, 'net_epoch%d.pth' % (epoch+1)))


if __name__ == "__main__":
    main()
