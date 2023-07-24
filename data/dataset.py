import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import random
import imagehash
from PIL import Image
from imageio import imread


class RandomRotate(object):
    def __call__(self, sample):
        k1 = np.random.randint(0, 4)
        sample['cl_img'] = np.rot90(sample['cl_img'], k1).copy()
        sample['rn_img'] = np.rot90(sample['rn_img'], k1).copy()
        k2 = np.random.randint(0, 4)
        sample['cl_ref'] = np.rot90(sample['cl_ref'], k2).copy()
        sample['rn_ref'] = np.rot90(sample['rn_ref'], k2).copy()
        return sample


class RandomFlip(object):
    def __call__(self, sample):
        if (np.random.randint(0, 2) == 1):
            sample['cl_img'] = np.fliplr(sample['cl_img']).copy()
            sample['rn_img'] = np.fliplr(sample['rn_img']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['cl_ref'] = np.fliplr(sample['cl_ref']).copy()
            sample['rn_ref'] = np.fliplr(sample['rn_ref']).copy()
        return sample


class ToTensor(object):
    def __call__(self, sample):
        cl_img, rn_img, cl_ref, rn_ref = sample['cl_img'], sample['rn_img'], sample['cl_ref'], sample['rn_ref']
        cl_img = cl_img.transpose((2, 0, 1))
        rn_img = rn_img.transpose((2, 0, 1))
        cl_ref = cl_ref.transpose((2, 0, 1))
        rn_ref = rn_ref.transpose((2, 0, 1))
        return {'cl_img': torch.from_numpy(cl_img).float(),
                'rn_img': torch.from_numpy(rn_img).float(),
                'cl_ref': torch.from_numpy(cl_ref).float(),
                'rn_ref': torch.from_numpy(rn_ref).float(),
                }


class derain_dataset(Dataset):
    def __init__(self, args, transform=transforms.Compose([])):
        super(derain_dataset, self).__init__()
        self.args = args
        self.rn_img = []
        self.cl_img = []
        self.rn_ref = []
        self.cl_ref = []
        self.imghash = []
        self.transform = transform

    def __len__(self):
        return len(self.rn_img)

    def get_imghash(self, imgs):
        imghash = []
        for i in range(len(imgs)):
            imghash.append(imagehash.phash(Image.open(imgs[i])))
        return imghash

    def retrieve_ref(self, idx, imghash):
        _imghash = imghash.copy()
        h = imagehash.phash(Image.open(self.cl_img[idx]))
        ref_dis = 64
        ref_idx = -1

        for i in range(len(_imghash)):
            if ((h - _imghash[i]) < ref_dis) & ((h - _imghash[i]) > 0):
                ref_idx = i
                ref_dis = h - _imghash[ref_idx]

        return ref_idx

    def __getitem__(self, idx):
        ref_idx = self.retrieve_ref(idx, self.imghash)

        ### clean img
        cl_img = imread(self.cl_img[idx])
        rn_img = imread(self.rn_img[idx])
        cl_ref = imread(self.cl_ref[ref_idx])
        rn_ref = imread(self.rn_ref[ref_idx])

        ### change type
        cl_img = cl_img.astype(np.float32)
        rn_img = rn_img.astype(np.float32)
        cl_ref = cl_ref.astype(np.float32)
        rn_ref = rn_ref.astype(np.float32)

        ### rgb range to [-1, 1]
        cl_img = (cl_img / 127.5) - 1.
        rn_img = (rn_img / 127.5) - 1.
        cl_ref = (cl_ref / 127.5) - 1.
        rn_ref = (rn_ref / 127.5) - 1.

        sample = {'cl_img': cl_img,
                  'rn_img': rn_img,
                  'cl_ref': cl_ref,
                  'rn_ref': rn_ref}

        if self.transform:
            sample = self.transform(sample)

        return sample


class BDD_train_set(derain_dataset):
    def __init__(self, args, transform=transforms.Compose([RandomFlip(), ToTensor()])):
        super(BDD_train_set, self).__init__(args)
        self.rn_img = sorted([os.path.join(args.dataset_dir, 'BDD100K/train/rainy', name) for name in
                              os.listdir(os.path.join(args.dataset_dir, 'BDD100K/train/rainy'))])
        self.cl_img = sorted([os.path.join(args.dataset_dir, 'BDD100K/train/clean', name) for name in
                              os.listdir(os.path.join(args.dataset_dir, 'BDD100K/train/clean'))])
        self.rn_ref = self.rn_img
        self.cl_ref = self.cl_img
        self.imghash = self.get_imghash(self.cl_ref)
        self.transform = transform


class BDD_val_set(derain_dataset):
    def __init__(self, args, transform=transforms.Compose([RandomFlip(), ToTensor()])):
        super(BDD_val_set, self).__init__(args)
        self.rn_img = sorted([os.path.join(args.dataset_dir, 'BDD100K/val/rainy', name) for name in
                              os.listdir(os.path.join(args.dataset_dir, 'BDD100K/val/rainy'))])
        self.cl_img = sorted([os.path.join(args.dataset_dir, 'BDD100K/val/clean', name) for name in
                              os.listdir(os.path.join(args.dataset_dir, 'BDD100K/val/clean'))])
        self.rn_ref = self.rn_img
        self.cl_ref = self.cl_img
        self.imghash = self.get_imghash(self.cl_ref)
        self.transform = transform


class BDD_test_set(derain_dataset):
    def __init__(self, args, transform=transforms.Compose([RandomFlip(), ToTensor()])):
        super(BDD_test_set, self).__init__(args)
        self.rn_img = sorted([os.path.join(args.dataset_dir, 'BDD100K/test/rainy', name) for name in
                              os.listdir(os.path.join(args.dataset_dir, 'BDD100K/test/rainy'))])
        self.cl_img = sorted([os.path.join(args.dataset_dir, 'BDD100K/test/clean', name) for name in
                              os.listdir(os.path.join(args.dataset_dir, 'BDD100K/test/clean'))])
        self.rn_ref = self.rn_img
        self.cl_ref = self.cl_img
        self.imghash = self.get_imghash(self.cl_ref)
        self.transform = transform


### use RainTrainL as reference dataset
class Rain100L(derain_dataset):
    def __init__(self, args, transform=transforms.Compose([ToTensor()])):
        super(Rain100L, self).__init__(args)
        self.rn_img = sorted([os.path.join(args.dataset_dir, 'Rain100L/rainy', name) for name in
                              os.listdir(os.path.join(args.dataset_dir, 'Rain100L/rainy'))])
        self.cl_img = sorted([os.path.join(args.dataset_dir, 'Rain100L/clean', name) for name in
                              os.listdir(os.path.join(args.dataset_dir, 'Rain100L/clean'))])
        self.rn_ref = sorted([os.path.join(args.dataset_dir, 'RainTrainL/rainy', name) for name in
                              os.listdir(os.path.join(args.dataset_dir, 'RainTrainL/rainy'))])
        self.cl_ref = sorted([os.path.join(args.dataset_dir, 'RainTrainL/clean', name) for name in
                              os.listdir(os.path.join(args.dataset_dir, 'RainTrainL/clean'))])
        self.imghash = self.get_imghash(self.cl_ref)
        self.transform = transform


### use RainTrainH as reference dataset
class Rain100H(derain_dataset):
    def __init__(self, args, transform=transforms.Compose([ToTensor()])):
        super(Rain100H, self).__init__(args)
        self.rn_img = sorted([os.path.join(args.dataset_dir, 'Rain100H/rainy', name) for name in
                              os.listdir(os.path.join(args.dataset_dir, 'Rain100H/rainy'))])
        self.cl_img = sorted([os.path.join(args.dataset_dir, 'Rain100H/clean', name) for name in
                              os.listdir(os.path.join(args.dataset_dir, 'Rain100H/clean'))])
        self.rn_ref = sorted([os.path.join(args.dataset_dir, 'RainTrainH/rainy', name) for name in
                              os.listdir(os.path.join(args.dataset_dir, 'RainTrainH/rainy'))])
        self.cl_ref = sorted([os.path.join(args.dataset_dir, 'RainTrainH/clean', name) for name in
                              os.listdir(os.path.join(args.dataset_dir, 'RainTrainH/clean'))])
        self.imghash = self.get_imghash(self.cl_ref)
        self.transform = transform


class RainTrainH_train(derain_dataset):
    def __init__(self, args, transform=transforms.Compose([RandomFlip(), RandomRotate(), ToTensor()]), split_ratio=0.9):
        super(RainTrainH_train, self).__init__(args)
        self.rn_img = sorted([os.path.join(args.dataset_dir, 'RainTrainH/rainy', name) for name in
                              os.listdir(os.path.join(args.dataset_dir, 'RainTrainH/rainy'))])
        self.cl_img = sorted([os.path.join(args.dataset_dir, 'RainTrainH/clean', name) for name in
                              os.listdir(os.path.join(args.dataset_dir, 'RainTrainH/clean'))])
        split_index = int(len(self.rn_img) * split_ratio)
        self.rn_img = self.rn_img[:split_index]
        self.cl_img = self.cl_img[:split_index]
        self.rn_ref = self.rn_img
        self.cl_ref = self.cl_img
        self.imghash = self.get_imghash(self.cl_ref)
        self.transform = transform


class RainTrainH_val(derain_dataset):
    def __init__(self, args, transform=transforms.Compose([RandomFlip(), RandomRotate(), ToTensor()]), split_ratio=0.9):
        super(RainTrainH_val, self).__init__(args)
        self.rn_img = sorted([os.path.join(args.dataset_dir, 'RainTrainH/rainy', name) for name in
                              os.listdir(os.path.join(args.dataset_dir, 'RainTrainH/rainy'))])
        self.cl_img = sorted([os.path.join(args.dataset_dir, 'RainTrainH/clean', name) for name in
                              os.listdir(os.path.join(args.dataset_dir, 'RainTrainH/clean'))])
        split_index = int(len(self.rn_img) * split_ratio)
        self.rn_img = self.rn_img[split_index:]
        self.cl_img = self.cl_img[split_index:]
        self.rn_ref = self.rn_img
        self.cl_ref = self.cl_img
        self.imghash = self.get_imghash(self.cl_ref)
        self.transform = transform


class RainTrainL_train(derain_dataset):
    def __init__(self, args, transform=transforms.Compose([RandomFlip(), RandomRotate(), ToTensor()]), split_ratio=0.9):
        super(RainTrainL_train, self).__init__(args)
        self.rn_img = sorted([os.path.join(args.dataset_dir, 'RainTrainL/rainy', name) for name in
                              os.listdir(os.path.join(args.dataset_dir, 'RainTrainL/rainy'))])
        self.cl_img = sorted([os.path.join(args.dataset_dir, 'RainTrainL/clean', name) for name in
                              os.listdir(os.path.join(args.dataset_dir, 'RainTrainL/clean'))])
        split_index = int(len(self.rn_img) * split_ratio)
        self.rn_img = self.rn_img[:split_index]
        self.cl_img = self.cl_img[:split_index]
        self.rn_ref = self.rn_img
        self.cl_ref = self.cl_img
        self.imghash = self.get_imghash(self.cl_ref)
        self.transform = transform


class RainTrainL_val(derain_dataset):
    def __init__(self, args, transform=transforms.Compose([RandomFlip(), RandomRotate(), ToTensor()]), split_ratio=0.9):
        super(RainTrainL_val, self).__init__(args)
        self.rn_img = sorted([os.path.join(args.dataset_dir, 'RainTrainL/rainy', name) for name in
                              os.listdir(os.path.join(args.dataset_dir, 'RainTrainL/rainy'))])
        self.cl_img = sorted([os.path.join(args.dataset_dir, 'RainTrainL/clean', name) for name in
                              os.listdir(os.path.join(args.dataset_dir, 'RainTrainL/clean'))])
        split_index = int(len(self.rn_img) * split_ratio)
        self.rn_img = self.rn_img[split_index:]
        self.cl_img = self.cl_img[split_index:]
        self.rn_ref = self.rn_img
        self.cl_ref = self.cl_img
        self.imghash = self.get_imghash(self.cl_ref)
        self.transform = transform


