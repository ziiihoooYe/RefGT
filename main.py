import os
import torch
import warnings


from option import args
from utils.utils import mkExpDir, load_model
from dataloader import get_dataloader
from model import DRTT
from loss.loss import get_loss_dict
from importlib import import_module
from trainer import Trainer
import utils.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

warnings.filterwarnings('ignore')

torch.manual_seed(1234)

os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'


if __name__ == '__main__':
    ### init DDP
    torch.multiprocessing.set_start_method('spawn')    
    dist.init()
    local_rank = dist.get_rank()
    ### make save_dir
    _logger = mkExpDir(args)
    torch.distributed.barrier()

    ### dataloader of training set and testing set
    _dataloader = get_dataloader(args)

    ### baseline model
    if args.baseline == 'PReNet':
        _baseline = getattr(import_module(args.baseline_module), 'PReNet')(recurrent_iter=6, use_GPU=True,
                                                                           device='cuda')
    else:
        _baseline = getattr(import_module(args.baseline_module), 'PReNet')(recurrent_iter=6, use_GPU=True,
                                                                           device='cuda')
    _baseline_device = torch.device('cuda', local_rank)
    _baseline.to(_baseline_device)
    _baseline = DDP(_baseline, device_ids=[local_rank], output_device=local_rank)
    load_model(_baseline, args.baseline_state_dir)  #load model state dict
    _baseline.eval()

    ###DRTT model
    _model = DRTT.DRTT(args)
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
        # t.compare_test()
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
            if epoch % args.val_every == 0:
                t.evaluate(current_epoch=epoch)
