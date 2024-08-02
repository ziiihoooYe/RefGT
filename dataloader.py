from torch.utils.data import DataLoader
from importlib import import_module
import torch.distributed as dist
import torch
import math
import os

def get_dataloader(args):
    ### import module
    dataset = import_module('data.dataset')

    ###construct dataset
    if args.dataset == 'BDD100K':
        data_train = getattr(dataset, 'BDD_train_set')(args)
        data_test = getattr(dataset, 'BDD_test_set')(args)
        data_val = getattr(dataset, 'BDD_val_set')(args)
    elif args.dataset == 'Rain100L':
        data_train = getattr(dataset, 'RainTrainL_train')(args)
        data_test = getattr(dataset, 'Rain100L')(args)
        data_val = getattr(dataset, 'RainTrainL_val')(args)
    elif args.dataset == 'Rain100H':
        data_train = getattr(dataset, 'RainTrainH_train')(args)
        data_val = getattr(dataset, 'RainTrainH_val')(args)
        data_test = getattr(dataset, 'Rain100H')(args)
    elif args.dataset == 'SPA-Data':
        data_train = getattr(dataset, 'SPAData_train')(args)
        data_val = getattr(dataset, 'SPAData_val')(args)
        data_test = getattr(dataset, 'SPAData_test')(args)
    elif args.dataset == 'KITTI':
        data_train = getattr(dataset, 'KITTI_train')(args)
        data_val = getattr(dataset, 'KITTI_val')(args)
        data_test = getattr(dataset, 'KITTI_test')(args)
    elif args.dataset == 'Cityscapes':
        data_train = getattr(dataset, 'Cityscapes_train')(args)
        data_val = getattr(dataset, 'Cityscapes_val')(args)
        data_test = getattr(dataset, 'Cityscapes_test')(args)
    else:
        raise SystemExit('Error: no such type of dataset!')
    
    if dist.is_initialized():
        #construct DDP smapler
        sampler_train = torch.utils.data.distributed.DistributedSampler(data_train)
        sampler_test = torch.utils.data.distributed.DistributedSampler(data_test)
        sampler_val = torch.utils.data.distributed.DistributedSampler(data_val)
            
        # construct dataloader
        dataloader_train = DataLoader(data_train, batch_size=args.batch_size,
                                    num_workers=args.num_workers, sampler=sampler_train)
        dataloader_test = DataLoader(data_test, batch_size=args.batch_size,
                                    num_workers=args.num_workers, sampler=sampler_test)
        dataloader_val = DataLoader(data_val, batch_size=args.batch_size,
                                    num_workers=args.num_workers, sampler=sampler_val)
    else:
        # baseline training (without ddp)
        dataloader_train = DataLoader(data_train, batch_size=args.batch_size,
                                    num_workers=args.num_workers)
        dataloader_test = DataLoader(data_test, batch_size=args.batch_size,
                                    num_workers=args.num_workers)
        dataloader_val = DataLoader(data_val, batch_size=args.batch_size,
                                    num_workers=args.num_workers)
    
    dataloader = {'train': dataloader_train, 'test': dataloader_test, 'val': dataloader_val}

    return dataloader
