import os
import torch
import warnings
from option import args
from utils.utils import mkExpDir, load_model, load_uformer
from dataloader import get_dataloader
from model import RefGT
from loss.loss import get_loss_dict
from importlib import import_module
from trainer import Trainer
import utils.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
warnings.filterwarnings('ignore')
torch.manual_seed(1234)

if __name__ == '__main__':
    ### init DDP
    torch.multiprocessing.set_start_method('spawn')    
    dist.init()
    local_rank = dist.get_rank()

    ### make save_dir
    _logger = mkExpDir(args)
    torch.distributed.barrier()

    ### dataloader 
    _dataloader = get_dataloader(args)

    ### load baseline model
    if args.baseline == 'PReNet':
        _baseline = getattr(import_module(args.baseline_module), 'PReNet')(recurrent_iter=6, use_GPU=True,
                                                                           device='cuda')
    elif args.baseline == 'GMM':
        _baseline = None
    elif args.baseline == 'Uformer':
        _baseline = getattr(import_module(args.baseline_module), 'Uformer')(img_size=128, embed_dim=32, win_size=8, 
                                                                            token_projection='linear',
                                                                            token_mlp='leff', depths=[1, 2, 8, 8, 2, 8, 8, 2, 1], modulator=True, dd_in=3)  
    else:
        _baseline = None
    
    # baseline model DDP
    if _baseline != None:
        _baseline_device = torch.device('cuda', local_rank)
        _baseline.to(_baseline_device)
        _baseline = DDP(_baseline, device_ids=[local_rank], output_device=local_rank)
        if args.baseline == 'Uformer':
            load_uformer(_baseline, args.baseline_state_dir)
        else:
            load_model(_baseline, args.baseline_state_dir)  #load model state dict
        _baseline.eval()

    ### load RefGT model
    _model = RefGT.RefGT(args)
    _model_device = torch.device('cuda', local_rank)
    _model.to(_model_device)
    _model = DDP(_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    ### loss
    _loss = get_loss_dict(args)

    ### trainer
    t = Trainer(args, _logger, _dataloader, _model, _loss, _baseline)

    ### test / eval / train
    if args.test:
        load_model(model=t.model, model_path=args.model_path)
        t.test()
    elif args.eval:
        load_model(model=t.model, model_path=args.model_path)
        t.evaluate()
    else:
        ### load pre-trained model
        if args.continue_training:
            load_model(model=t.model, model_path=args.model_path)

        for epoch in range(1, args.num_epochs + 1):
            for dataloader in t.dataloader.values():
                dataloader.sampler.set_epoch(epoch)
            
            if epoch < args.gt_init_epochs and not args.continue_training:
                ### ground truth initialization training
                t.train(current_epoch=epoch, is_init=True)
            else:
                ### training
                t.train(current_epoch=epoch, is_init=False)

            ### evaluation
            if epoch % args.val_every_epoch == 0:
                t.evaluate(current_epoch=epoch)
