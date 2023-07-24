import os

from option import args
from utils import mkExpDir, load_model
from dataloader import get_dataloader
from model import DRTT
from loss.loss import get_loss_dict
from trainer import Trainer
from importlib import import_module

import torch
import torch.nn as nn
import warnings

warnings.filterwarnings('ignore')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3, 4, 5, 6, 7, 0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

torch.manual_seed(1234)


if __name__ == '__main__':
    ### make save_dir
    _logger = mkExpDir(args)

    ### dataloader of training set and testing set
    _dataloader = get_dataloader(args)

    ### baseline model
    if args.backbone == 'PReNet':
        _backbone = getattr(import_module(args.backbone_module), 'PReNet')(recurrent_iter=6, use_GPU=True,
                                                                           device=args.backbone_device)
    else:
        _backbone = getattr(import_module(args.backbone_module), 'PReNet')(recurrent_iter=6, use_GPU=True,
                                                                           device=args.backbone_device)
    _backbone_dir = os.path.join(args.backbone_state_dir, 'net_latest.pth')
    load_model(_backbone, _backbone_dir)
    backbone_device = torch.device('cpu' if args.cpu else args.backbone_device)
    _backbone.to(backbone_device)
    _backbone.eval()

    ### device and model
    device = torch.device('cpu' if args.cpu else 'cuda')
    _model = DRTT.DRTT(args)
    if (not args.cpu) and (args.num_gpu > 1):
        _model = nn.DataParallel(_model, list(range(args.num_gpu)))
    _model.to(device)

    ### loss
    _loss = get_loss_dict(args)

    ### trainer
    t = Trainer(args, _logger, _dataloader, _model, _loss, _backbone)

    ### test / eval / train
    if args.test:
        # t.load(model_path=args.model_path)
        load_model(t.model, model_path=args.model_path, data_parallel=((not args.cpu) and (args.num_gpu > 1)))
        t.test()
        # t.compare_test()
    elif args.eval:
        # t.load(model_path=args.model_path)
        load_model(t.model, model_path=args.model_path, data_parallel=((not args.cpu) and (args.num_gpu > 1)))
        t.evaluate()
    else:
        ### load pre-trained model
        if args.continue_training:
            # t.load(model_path=args.model_path)
            load_model(t.model, model_path=args.model_path, data_parallel=((not args.cpu) and (args.num_gpu > 1)))

        for epoch in range(1, args.num_epochs + 1):
            if epoch < args.gt_init_epochs and not args.continue_training:
                ### ground truth initialization training
                t.args.ms_ssim_l1_loss = False
                t.train(current_epoch=epoch, gt_ref=True)
            else:
                ### training
                t.args.ms_ssim_l1_loss = True
                t.train(current_epoch=epoch, gt_ref=False)

            ### evaluation
            if epoch % args.val_every == 0:
                t.evaluate(current_epoch=epoch)
