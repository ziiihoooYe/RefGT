import os
import re
import torch
import cv2
from overrides import overrides
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import random
import imagehash
from PIL import Image
from imageio import imread
import time


class RandomRotate(object):
    def __call__(self, sample):
        angle = np.random.randint(0, 4)
        for k in list(sample.keys()):
            sample[k] = np.rot90(sample[k], angle).copy()
        return sample


class ResizeImage(object):
    def __init__(self, height, width, interpolation=cv2.INTER_LINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, sample):
        # Resizing the images to the specified dimensions
        for k in list(sample.keys()):
            sample[k] = cv2.resize(
                sample[k], (self.width, self.height), interpolation=self.interpolation)
        return sample


class RandomFlip(object):
    def __call__(self, sample):
        if (np.random.randint(0, 2) == 1):
            for k in list(sample.keys()):
                sample[k] = np.fliplr(sample[k]).copy()
        return sample


class ToTensor(object):
    def __call__(self, sample):
        for k in list(sample.keys()):
            sample[k] = torch.from_numpy(
                sample[k].transpose((2, 0, 1))).float()
        return sample


class RemoveAlphaChannel(object):
    def __call__(self, sample):
        for k in list(sample.keys()):
            sample[k] = sample[k][:, :, :3]
        return sample


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

        if args.baseline == 'GMM':
            self.gmm_img = []
            self.gmm_ref = []

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

    def split_img(self, img, n):
        split_img = torch.split(img, 256, dim=-1)
        return split_img[n]

    def __getitem__(self, idx):
        start_time = time.time()
        ref_idx = self.retrieve_ref(idx, self.imghash)
        retrieval_time = time.time() - start_time

        # clean img
        cl_img = imread(self.cl_img[idx])
        rn_img = imread(self.rn_img[idx])
        cl_ref = imread(self.cl_ref[ref_idx])
        rn_ref = imread(self.rn_ref[ref_idx])

        # change type
        cl_img = cl_img.astype(np.float32)
        rn_img = rn_img.astype(np.float32)
        cl_ref = cl_ref.astype(np.float32)
        rn_ref = rn_ref.astype(np.float32)

        # rgb range to [-1, 1]
        cl_img = (cl_img / 127.5) - 1.
        rn_img = (rn_img / 127.5) - 1.
        cl_ref = (cl_ref / 127.5) - 1.
        rn_ref = (rn_ref / 127.5) - 1.

        sample = {'cl_img': cl_img,
                  'rn_img': rn_img,
                  'cl_ref': cl_ref,
                  'rn_ref': rn_ref}

        if self.args.baseline == 'GMM':
            gmm_img = imread(self.gmm_img[idx])
            gmm_ref = imread(self.gmm_ref[ref_idx])

            gmm_img = gmm_img.astype(np.float32)
            gmm_ref = gmm_ref.astype(np.float32)

            gmm_img = (gmm_img / 127.5) - 1.
            gmm_ref = (gmm_ref / 127.5) - 1.

            sample['gmm_img'] = gmm_img
            sample['gmm_ref'] = gmm_ref

        if self.transform:
            sample = self.transform(sample)

        if self.args.resize_img:
            n = random.randint(0, 3)
            for k in list(sample.keys()):
                sample[k] = self.split_img(sample[k], n).clone()

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

        if args.baseline == 'GMM':
            self.gmm_img = sorted([os.path.join(args.dataset_dir, 'BDD100K_GMM/train/rainy', name) for name in
                                   os.listdir(os.path.join(args.dataset_dir, 'BDD100K_GMM/train/rainy'))])
            self.gmm_ref = self.gmm_img

            assert len(self.gmm_img) == len(self.rn_img)
            assert len(self.gmm_ref) == len(self.rn_ref)


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

        if args.baseline == 'GMM':
            self.gmm_img = sorted([os.path.join(args.dataset_dir, 'BDD100K_GMM/val/rainy', name) for name in
                                   os.listdir(os.path.join(args.dataset_dir, 'BDD100K_GMM/val/rainy'))])
            self.gmm_ref = self.gmm_img

            assert len(self.gmm_img) == len(self.rn_img)
            assert len(self.gmm_ref) == len(self.rn_ref)


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

        if args.baseline == 'GMM':
            self.gmm_img = sorted([os.path.join(args.dataset_dir, 'BDD100K_GMM/test/rainy', name) for name in
                                   os.listdir(os.path.join(args.dataset_dir, 'BDD100K_GMM/test/rainy'))])
            self.gmm_ref = self.gmm_img

            assert len(self.gmm_img) == len(self.rn_img)
            assert len(self.gmm_ref) == len(self.rn_ref)


class SPAData(derain_dataset):
    def __init__(self, args):
        super(SPAData, self).__init__(args)

    def retrieve_ref(self, idx, img_hash):
        _imghash = img_hash.copy()
        h = imagehash.phash(Image.open(self.cl_img[idx]))
        ref_dis = 64
        ref_idx = -1

        for i in range(len(_imghash)):
            if ((h - _imghash[i]) < ref_dis) and ((h - _imghash[i]) > 0) and (self.allow_same_video or not self.is_same_video(self.cl_img[idx], self.cl_ref[i])):
                ref_idx = i
                ref_dis = h - _imghash[ref_idx]

        return ref_idx

    def is_same_video(self, img, ref):
        pattern = r"(\d+)_(\d+_\d+)\.png"

        img_video = re.match(pattern, os.path.basename(img)).group(1)
        ref_video = re.match(pattern, os.path.basename(ref)).group(1)

        return (img_video == ref_video)


class SPAData_train(SPAData):
    def __init__(self, args, allow_same_video=False, split_ratio=0.8, transform=transforms.Compose([ToTensor()])):
        super().__init__(args)
        self.allow_same_video = allow_same_video
        self.split_ratio = split_ratio
        self.rn_img = sorted([os.path.join(args.dataset_dir, 'SPA-Data/train/rainy', video, name)
                              for video in os.listdir(os.path.join(args.dataset_dir, 'SPA-Data/train/rainy'))
                              for name in os.listdir(os.path.join(args.dataset_dir, 'SPA-Data/train/rainy', video))])
        self.cl_img = sorted([os.path.join(args.dataset_dir, 'SPA-Data/train/clean', video, name)
                              for video in os.listdir(os.path.join(args.dataset_dir, 'SPA-Data/train/clean'))
                              for name in os.listdir(os.path.join(args.dataset_dir, 'SPA-Data/train/clean', video))])
        assert len(self.rn_img) == len(
            self.cl_img), 'rainy image do not match clean image'
        self.rn_ref = self.rn_img
        self.cl_ref = self.cl_img
        self.imghash = self.get_imghash(self.cl_ref)
        self.transform = transform

        split_index = int(len(self.rn_img) * self.split_ratio)
        self.rn_img = self.rn_img[:split_index]
        self.cl_img = self.cl_img[:split_index]

        if args.baseline == 'GMM':
            self.gmm_img = sorted([os.path.join(args.dataset_dir, 'SPA-Data-GMM/GMM', video, name)
                                  for video in os.listdir(os.path.join(args.dataset_dir, 'SPA-Data-GMM/GMM'))
                                  for name in os.listdir(os.path.join(args.dataset_dir, 'SPA-Data-GMM/GMM', video))])
            self.gmm_ref = self.gmm_img

            self.gmm_img = self.gmm_img[:split_index]

            assert len(self.gmm_img) == len(self.rn_img)
            assert len(self.gmm_ref) == len(self.rn_ref)


class SPAData_val(SPAData):
    def __init__(self, args, allow_same_video=False, split_ratio=0.8, transform=transforms.Compose([ToTensor()])):
        super().__init__(args)
        self.allow_same_video = allow_same_video
        self.split_ratio = split_ratio
        self.rn_img = sorted([os.path.join(args.dataset_dir, 'SPA-Data/train/rainy', video, name)
                              for video in os.listdir(os.path.join(args.dataset_dir, 'SPA-Data/train/rainy'))
                              for name in os.listdir(os.path.join(args.dataset_dir, 'SPA-Data/train/rainy', video))])
        self.cl_img = sorted([os.path.join(args.dataset_dir, 'SPA-Data/train/clean', video, name)
                              for video in os.listdir(os.path.join(args.dataset_dir, 'SPA-Data/train/clean'))
                              for name in os.listdir(os.path.join(args.dataset_dir, 'SPA-Data/train/clean', video))])
        assert len(self.rn_img) == len(
            self.cl_img), 'rainy image do not match clean image'
        self.rn_ref = self.rn_img
        self.cl_ref = self.cl_img
        self.imghash = self.get_imghash(self.cl_ref)
        self.transform = transform

        split_index = int(len(self.rn_img) * self.split_ratio)
        stop_ratio = (1 - (1 - self.split_ratio) / 2)
        stop_index = int(len(self.rn_img) * stop_ratio)
        self.rn_img = self.rn_img[split_index:stop_index]
        self.cl_img = self.cl_img[split_index:stop_index]

        if args.baseline == 'GMM':
            self.gmm_img = sorted([os.path.join(args.dataset_dir, 'SPA-Data-GMM/GMM', video, name)
                                  for video in os.listdir(os.path.join(args.dataset_dir, 'SPA-Data-GMM/GMM'))
                                  for name in os.listdir(os.path.join(args.dataset_dir, 'SPA-Data-GMM/GMM', video))])
            self.gmm_ref = self.gmm_img

            self.gmm_img = self.gmm_img[split_index:stop_index]

            assert len(self.gmm_img) == len(self.rn_img)
            assert len(self.gmm_ref) == len(self.rn_ref)


class SPAData_test(SPAData):
    def __init__(self, args, allow_same_video=False, split_ratio=0.8, transform=transforms.Compose([ToTensor()])):
        super().__init__(args)
        self.allow_same_video = allow_same_video
        self.split_ratio = split_ratio
        self.rn_img = sorted([os.path.join(args.dataset_dir, 'SPA-Data/train/rainy', video, name)
                              for video in os.listdir(os.path.join(args.dataset_dir, 'SPA-Data/train/rainy'))
                              for name in os.listdir(os.path.join(args.dataset_dir, 'SPA-Data/train/rainy', video))])
        self.cl_img = sorted([os.path.join(args.dataset_dir, 'SPA-Data/train/clean', video, name)
                              for video in os.listdir(os.path.join(args.dataset_dir, 'SPA-Data/train/clean'))
                              for name in os.listdir(os.path.join(args.dataset_dir, 'SPA-Data/train/clean', video))])
        assert len(self.rn_img) == len(
            self.cl_img), 'rainy image do not match clean image'
        self.rn_ref = self.rn_img
        self.cl_ref = self.cl_img
        self.imghash = self.get_imghash(self.cl_ref)
        self.transform = transform

        stop_ratio = (1 - (1 - self.split_ratio) / 2)
        stop_index = int(len(self.rn_img) * stop_ratio)
        self.rn_img = self.rn_img[stop_index:]
        self.cl_img = self.cl_img[stop_index:]

        if args.baseline == 'GMM':
            self.gmm_img = sorted([os.path.join(args.dataset_dir, 'SPA-Data-GMM/GMM', video, name)
                                  for video in os.listdir(os.path.join(args.dataset_dir, 'SPA-Data-GMM/GMM'))
                                  for name in os.listdir(os.path.join(args.dataset_dir, 'SPA-Data-GMM/GMM', video))])
            self.gmm_ref = self.gmm_img

            self.gmm_img = self.gmm_img[stop_index:]

            assert len(self.gmm_img) == len(self.rn_img)
            assert len(self.gmm_ref) == len(self.rn_ref)


class KITTI_train(derain_dataset):
    def __init__(self, args, transform=transforms.Compose([RemoveAlphaChannel(), ResizeImage(256, 1024), ToTensor()]), split_ratio=0.8):
        super().__init__(args)
        self.split_ratio = split_ratio
        if args.baseline != 'GMM':
            self.rn_img = sorted([os.path.join(args.dataset_dir, 'KITTI_Rain/rain', name) for name in
                                  os.listdir(os.path.join(args.dataset_dir, 'KITTI_Rain/rain'))])
            self.cl_img = sorted([os.path.join(args.dataset_dir, 'KITTI_Rain/clean', name) for name in
                                  os.listdir(os.path.join(args.dataset_dir, 'KITTI_Rain/clean'))])
        else:
            self.rn_img = sorted([os.path.join(args.dataset_dir, 'KITTI-GMM/rain', name) for name in
                                  os.listdir(os.path.join(args.dataset_dir, 'KITTI-GMM/rain'))])
            self.cl_img = sorted([os.path.join(args.dataset_dir, 'KITTI-GMM/clean', name) for name in
                                  os.listdir(os.path.join(args.dataset_dir, 'KITTI-GMM/clean'))])

        self.rn_ref = self.rn_img
        self.cl_ref = self.cl_img
        self.imghash = self.get_imghash(self.cl_ref)
        self.transform = transform

        split_index = int(len(self.rn_img) * self.split_ratio)
        self.rn_img = self.rn_img[:split_index]
        self.cl_img = self.cl_img[:split_index]

        if args.baseline == 'GMM':
            self.gmm_img = sorted([os.path.join(args.dataset_dir, 'KITTI-GMM/GMM', name)
                                  for name in os.listdir(os.path.join(args.dataset_dir, 'KITTI-GMM/GMM'))])
            self.gmm_ref = self.gmm_img

            self.gmm_img = self.gmm_img[:split_index]

            assert len(self.gmm_img) == len(self.rn_img)
            assert len(self.gmm_ref) == len(self.rn_ref)


class KITTI_val(derain_dataset):
    def __init__(self, args, transform=transforms.Compose([RemoveAlphaChannel(), ResizeImage(256, 1024), ToTensor()]), split_ratio=0.8):
        super().__init__(args)
        self.split_ratio = split_ratio
        if args.baseline != 'GMM':
            self.rn_img = sorted([os.path.join(args.dataset_dir, 'KITTI_Rain/rain', name) for name in
                                  os.listdir(os.path.join(args.dataset_dir, 'KITTI_Rain/rain'))])
            self.cl_img = sorted([os.path.join(args.dataset_dir, 'KITTI_Rain/clean', name) for name in
                                  os.listdir(os.path.join(args.dataset_dir, 'KITTI_Rain/clean'))])
        else:
            self.rn_img = sorted([os.path.join(args.dataset_dir, 'KITTI-GMM/rain', name) for name in
                                  os.listdir(os.path.join(args.dataset_dir, 'KITTI-GMM/rain'))])
            self.cl_img = sorted([os.path.join(args.dataset_dir, 'KITTI-GMM/clean', name) for name in
                                  os.listdir(os.path.join(args.dataset_dir, 'KITTI-GMM/clean'))])
        self.rn_ref = self.rn_img
        self.cl_ref = self.cl_img
        self.imghash = self.get_imghash(self.cl_ref)
        self.transform = transform

        split_index = int(len(self.rn_img) * self.split_ratio)
        stop_ratio = (1 - (1 - self.split_ratio) / 2)
        stop_index = int(len(self.rn_img) * stop_ratio)
        self.rn_img = self.rn_img[split_index:stop_index]
        self.cl_img = self.cl_img[split_index:stop_index]

        if args.baseline == 'GMM':
            self.gmm_img = sorted([os.path.join(args.dataset_dir, 'KITTI-GMM/GMM', name)
                                  for name in os.listdir(os.path.join(args.dataset_dir, 'KITTI-GMM/GMM'))])
            self.gmm_ref = self.gmm_img

            self.gmm_img = self.gmm_img[split_index:stop_index]

            assert len(self.gmm_img) == len(self.rn_img)
            assert len(self.gmm_ref) == len(self.rn_ref)


class KITTI_test(derain_dataset):
    def __init__(self, args, transform=transforms.Compose([RemoveAlphaChannel(), ResizeImage(256, 1024), ToTensor()]), split_ratio=0.8):
        super().__init__(args)
        self.split_ratio = split_ratio
        if args.baseline != 'GMM':
            self.rn_img = sorted([os.path.join(args.dataset_dir, 'KITTI_Rain/rain', name) for name in
                                  os.listdir(os.path.join(args.dataset_dir, 'KITTI_Rain/rain'))])
            self.cl_img = sorted([os.path.join(args.dataset_dir, 'KITTI_Rain/clean', name) for name in
                                  os.listdir(os.path.join(args.dataset_dir, 'KITTI_Rain/clean'))])
        else:
            self.rn_img = sorted([os.path.join(args.dataset_dir, 'KITTI-GMM/rain', name) for name in
                                  os.listdir(os.path.join(args.dataset_dir, 'KITTI-GMM/rain'))])
            self.cl_img = sorted([os.path.join(args.dataset_dir, 'KITTI-GMM/clean', name) for name in
                                  os.listdir(os.path.join(args.dataset_dir, 'KITTI-GMM/clean'))])
        self.rn_ref = self.rn_img
        self.cl_ref = self.cl_img
        self.imghash = self.get_imghash(self.cl_ref)
        self.transform = transform

        stop_ratio = (1 - (1 - self.split_ratio) / 2)
        stop_index = int(len(self.rn_img) * stop_ratio)
        self.rn_img = self.rn_img[stop_index:]
        self.cl_img = self.cl_img[stop_index:]

        if args.baseline == 'GMM':
            self.gmm_img = sorted([os.path.join(args.dataset_dir, 'KITTI-GMM/GMM', name)
                                  for name in os.listdir(os.path.join(args.dataset_dir, 'KITTI-GMM/GMM'))])
            self.gmm_ref = self.gmm_img

            self.gmm_img = self.gmm_img[stop_index:]

            assert len(self.gmm_img) == len(self.rn_img)
            assert len(self.gmm_ref) == len(self.rn_ref)


class Cityscapes_train(derain_dataset):
    def __init__(self, args, transform=transforms.Compose([ResizeImage(256, 1024), ToTensor()]), split_ratio=0.9):
        super().__init__(args)
        self.split_ratio = split_ratio
        if args.baseline != 'GMM':
            self.rn_img = sorted([os.path.join(args.dataset_dir, 'Cityscapes_Rain/train/rain', city, name)
                                  for city in os.listdir(os.path.join(args.dataset_dir, 'Cityscapes_Rain/train/rain'))
                                  for name in os.listdir(os.path.join(args.dataset_dir, 'Cityscapes_Rain/train/rain', city))])
            self.cl_img = sorted([os.path.join(args.dataset_dir, 'Cityscapes_Rain/train/clean', city, name)
                                  for city in os.listdir(os.path.join(args.dataset_dir, 'Cityscapes_Rain/train/clean'))
                                  for name in os.listdir(os.path.join(args.dataset_dir, 'Cityscapes_Rain/train/clean', city))])
        else:
            self.rn_img = sorted([os.path.join(args.dataset_dir, 'Cityscapes-GMM/train/rain', city, name)
                                  for city in os.listdir(os.path.join(args.dataset_dir, 'Cityscapes-GMM/train/rain'))
                                  for name in os.listdir(os.path.join(args.dataset_dir, 'Cityscapes-GMM/train/rain', city))])
            self.cl_img = sorted([os.path.join(args.dataset_dir, 'Cityscapes-GMM/train/clean', city, name)
                                  for city in os.listdir(os.path.join(args.dataset_dir, 'Cityscapes-GMM/train/clean'))
                                  for name in os.listdir(os.path.join(args.dataset_dir, 'Cityscapes-GMM/train/clean', city))])

        self.rn_ref = self.rn_img
        self.cl_ref = self.cl_img
        self.imghash = self.get_imghash(self.cl_ref)
        self.transform = transform

        split_index = int(len(self.rn_img) * self.split_ratio)
        self.rn_img = self.rn_img[:split_index]
        self.cl_img = self.cl_img[:split_index]

        if args.baseline == 'GMM':
            self.gmm_img = sorted([os.path.join(args.dataset_dir, 'Cityscapes-GMM/train/GMM', city, name)
                                   for city in os.listdir(os.path.join(args.dataset_dir, 'Cityscapes-GMM/train/GMM'))
                                   for name in os.listdir(os.path.join(args.dataset_dir, 'Cityscapes-GMM/train/GMM', city))])
            self.gmm_ref = self.gmm_img

            self.gmm_img = self.gmm_img[:split_index]

            assert len(self.gmm_img) == len(self.rn_img)
            assert len(self.gmm_ref) == len(self.rn_ref)


class Cityscapes_val(derain_dataset):
    def __init__(self, args, transform=transforms.Compose([ResizeImage(256, 1024), ToTensor()]), split_ratio=0.9):
        super().__init__(args)
        self.split_ratio = split_ratio
        if args.baseline != 'GMM':
            self.rn_img = sorted([os.path.join(args.dataset_dir, 'Cityscapes_Rain/train/rain', city, name)
                                  for city in os.listdir(os.path.join(args.dataset_dir, 'Cityscapes_Rain/train/rain'))
                                  for name in os.listdir(os.path.join(args.dataset_dir, 'Cityscapes_Rain/train/rain', city))])
            self.cl_img = sorted([os.path.join(args.dataset_dir, 'Cityscapes_Rain/train/clean', city, name)
                                  for city in os.listdir(os.path.join(args.dataset_dir, 'Cityscapes_Rain/train/clean'))
                                  for name in os.listdir(os.path.join(args.dataset_dir, 'Cityscapes_Rain/train/clean', city))])
        else:
            self.rn_img = sorted([os.path.join(args.dataset_dir, 'Cityscapes-GMM/train/rain', city, name)
                                  for city in os.listdir(os.path.join(args.dataset_dir, 'Cityscapes-GMM/train/rain'))
                                  for name in os.listdir(os.path.join(args.dataset_dir, 'Cityscapes-GMM/train/rain', city))])
            self.cl_img = sorted([os.path.join(args.dataset_dir, 'Cityscapes-GMM/train/clean', city, name)
                                  for city in os.listdir(os.path.join(args.dataset_dir, 'Cityscapes-GMM/train/clean'))
                                  for name in os.listdir(os.path.join(args.dataset_dir, 'Cityscapes-GMM/train/clean', city))])
        self.rn_ref = self.rn_img
        self.cl_ref = self.cl_img
        self.imghash = self.get_imghash(self.cl_ref)
        self.transform = transform

        split_index = int(len(self.rn_img) * self.split_ratio)
        self.rn_img = self.rn_img[split_index:]
        self.cl_img = self.cl_img[split_index:]

        if args.baseline == 'GMM':
            self.gmm_img = sorted([os.path.join(args.dataset_dir, 'Cityscapes-GMM/train/GMM', city, name)
                                   for city in os.listdir(os.path.join(args.dataset_dir, 'Cityscapes-GMM/train/GMM'))
                                   for name in os.listdir(os.path.join(args.dataset_dir, 'Cityscapes-GMM/train/GMM', city))])
            self.gmm_ref = self.gmm_img

            self.gmm_img = self.gmm_img[split_index:]

            assert len(self.gmm_img) == len(self.rn_img)
            assert len(self.gmm_ref) == len(self.rn_ref)


class Cityscapes_test(derain_dataset):
    def __init__(self, args, transform=transforms.Compose([ResizeImage(256, 1024), ToTensor()])):
        super().__init__(args)
        if args.baseline != 'GMM':
            self.rn_img = sorted([os.path.join(args.dataset_dir, 'Cityscapes_Rain/val/rain', city, name)
                                  for city in os.listdir(os.path.join(args.dataset_dir, 'Cityscapes_Rain/val/rain'))
                                  for name in os.listdir(os.path.join(args.dataset_dir, 'Cityscapes_Rain/val/rain', city))])
            self.cl_img = sorted([os.path.join(args.dataset_dir, 'Cityscapes_Rain/val/clean', city, name)
                                  for city in os.listdir(os.path.join(args.dataset_dir, 'Cityscapes_Rain/val/clean'))
                                  for name in os.listdir(os.path.join(args.dataset_dir, 'Cityscapes_Rain/val/clean', city))])
        else:
            self.rn_img = sorted([os.path.join(args.dataset_dir, 'Cityscapes-GMM/val/rain', city, name)
                                  for city in os.listdir(os.path.join(args.dataset_dir, 'Cityscapes-GMM/val/rain'))
                                  for name in os.listdir(os.path.join(args.dataset_dir, 'Cityscapes-GMM/val/rain', city))])
            self.cl_img = sorted([os.path.join(args.dataset_dir, 'Cityscapes-GMM/val/clean', city, name)
                                  for city in os.listdir(os.path.join(args.dataset_dir, 'Cityscapes-GMM/val/clean'))
                                  for name in os.listdir(os.path.join(args.dataset_dir, 'Cityscapes-GMM/val/clean', city))])
        self.rn_ref = self.rn_img
        self.cl_ref = self.cl_img
        self.imghash = self.get_imghash(self.cl_ref)
        self.transform = transform

        if args.baseline == 'GMM':
            self.gmm_img = sorted([os.path.join(args.dataset_dir, 'Cityscapes-GMM/val/GMM', city, name)
                                   for city in os.listdir(os.path.join(args.dataset_dir, 'Cityscapes-GMM/val/GMM'))
                                   for name in os.listdir(os.path.join(args.dataset_dir, 'Cityscapes-GMM/val/GMM', city))])
            self.gmm_ref = self.gmm_img

            assert len(self.gmm_img) == len(self.rn_img)
            assert len(self.gmm_ref) == len(self.rn_ref)
