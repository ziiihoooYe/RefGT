from torch.utils.data import DataLoader
from importlib import import_module
import torch
import math
import os

from torchvision.transforms.functional import pad


def pad_image_to_size(img, target_height, target_width):
    h, w = img.shape[1], img.shape[2]
    pad_top = target_height - h
    pad_left = target_width - w
    padding = [pad_left, pad_top, 0, 0]
    return pad(img, padding, fill=0, padding_mode='reflect')


### pad img in the same batch to the same size (for data parallel)
def custom_collate_fn(batch):
    cl_imgs = [item['cl_img'] for item in batch]
    rn_imgs = [item['rn_img'] for item in batch]
    cl_refs = [item['cl_ref'] for item in batch]
    rn_refs = [item['rn_ref'] for item in batch]

    # obtain maximum
    max_h = max([img.shape[1] for img in cl_imgs + cl_refs])
    max_w = max([img.shape[2] for img in cl_imgs + cl_refs])
    max_h = ((math.floor(max_h / 4)) + 1) * 4
    max_w = ((math.floor(max_w / 4)) + 1) * 4

    # padding img to the maximum size
    cl_imgs_padded = [pad_image_to_size(img, max_h, max_w) for img in cl_imgs]
    rn_imgs_padded = [pad_image_to_size(img, max_h, max_w) for img in rn_imgs]
    cl_refs_padded = [pad_image_to_size(img, max_h, max_w) for img in cl_refs]
    rn_refs_padded = [pad_image_to_size(img, max_h, max_w) for img in rn_refs]

    # stack
    cl_imgs_tensor = torch.stack(cl_imgs_padded, 0)
    rn_imgs_tensor = torch.stack(rn_imgs_padded, 0)
    cl_refs_tensor = torch.stack(cl_refs_padded, 0)
    rn_refs_tensor = torch.stack(rn_refs_padded, 0)

    # store size
    cl_imgs_sizes = [(img.shape[1], img.shape[2]) for img in cl_imgs]
    rn_imgs_sizes = [(img.shape[1], img.shape[2]) for img in rn_imgs]
    cl_refs_sizes = [(img.shape[1], img.shape[2]) for img in cl_refs]
    rn_refs_sizes = [(img.shape[1], img.shape[2]) for img in rn_refs]

    # 返回一个字典，包含经过 padding 的图像张量和原始图像的大小
    return {'cl_img': cl_imgs_tensor, 'rn_img': rn_imgs_tensor, 'cl_ref': cl_refs_tensor, 'rn_ref': rn_refs_tensor,
            'cl_img_sizes': cl_imgs_sizes, 'rn_img_sizes': rn_imgs_sizes,
            'cl_ref_sizes': cl_refs_sizes, 'rn_ref_sizes': rn_refs_sizes}



def get_dataloader(args):
    ### import module
    m = import_module('data.dataset')

    if args.dataset == 'BDD100K':
        data_train = getattr(m, 'BDD_train_set')(args)
        data_test = getattr(m, 'BDD_val_set')(args)
        data_val = getattr(m, 'BDD_test_set')(args)
        dataloader_train = DataLoader(data_train, batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.num_workers, collate_fn=custom_collate_fn)
        dataloader_val = DataLoader(data_test, batch_size=args.batch_size,
                                    shuffle=True, num_workers=args.num_workers, collate_fn=custom_collate_fn)
        dataloader_test = DataLoader(data_val, batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.num_workers, collate_fn=custom_collate_fn)
        dataloader = {'train': dataloader_train, 'test': dataloader_test, 'val': dataloader_val}
    elif args.dataset == 'Rain100L':
        data_train = getattr(m, 'RainTrainL_train')(args)
        data_val = getattr(m, 'RainTrainL_val')(args)
        data_test = getattr(m, 'Rain100L')(args)
        dataloader_train = DataLoader(data_train, batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.num_workers, collate_fn=custom_collate_fn)
        dataloader_val = DataLoader(data_val, batch_size=args.batch_size,
                                    shuffle=True, num_workers=args.num_workers, collate_fn=custom_collate_fn)
        dataloader_test = DataLoader(data_test, batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.num_workers, collate_fn=custom_collate_fn)
        dataloader = {'train': dataloader_train, 'test': dataloader_test, 'val': dataloader_val}
    elif args.dataset == 'Rain100H':
        data_train = getattr(m, 'RainTrainH_train')(args)
        data_val = getattr(m, 'RainTrainH_val')(args)
        data_test = getattr(m, 'Rain100H')(args)
        dataloader_train = DataLoader(data_train, batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.num_workers, collate_fn=custom_collate_fn)
        dataloader_val = DataLoader(data_test, batch_size=args.batch_size,
                                    shuffle=True, num_workers=args.num_workers, collate_fn=custom_collate_fn)
        dataloader_test = DataLoader(data_val, batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.num_workers, collate_fn=custom_collate_fn)
        dataloader = {'train': dataloader_train, 'test': dataloader_test, 'val': dataloader_val}
    else:
        raise SystemExit('Error: no such type of dataset!')

    return dataloader
