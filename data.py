import os, glob, random
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from joint_transforms import Compose, Resize, RandomHorizontallyFlip

import cv2

class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize, is_agu=False):
        self.trainsize = trainsize
        self.agu = is_agu
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]

        skeleton_root = '/userhome/data/SOD/DUTS_TRAIN/skeleton_gts/'
        edge_root = '/userhome/data/SOD/DUTS_TRAIN/edge_gts/'

        self.skeleton_gts = [skeleton_root + f for f in os.listdir(skeleton_root) if f.endswith('.jpg')
                             or f.endswith('.png')]
        self.edge_gts = [edge_root + f for f in os.listdir(edge_root) if f.endswith('.jpg')
                         or f.endswith('.png')]

        if self.agu:
            self.joint_transform = Compose([
                Resize(self.trainsize),
                RandomHorizontallyFlip(),
            ])
        else:
            self.joint_transform = Resize(self.trainsize)

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.skeleton_gts = sorted(self.skeleton_gts)
        self.edge_gts = sorted(self.edge_gts)
        self.size = len(self.images)

    def __getitem__(self, index):
        img, gt = self.rgb_loader(self.images[index]), self.binary_loader(self.gts[index])
        skeleton, edge = self.binary_loader(self.skeleton_gts[index]), self.binary_loader(self.edge_gts[index])
        img, gt, skeleton, edge = self.joint_transform([img, gt, skeleton, edge])

        img, gt = np.array(img, dtype=np.float32), np.array(gt, dtype=np.float32)
        skeleton, edge = np.array(skeleton, dtype=np.float32), np.array(edge, dtype=np.float32)
        img = img[:, :, ::-1].copy()
        img -= np.array((104.00699, 116.66877, 122.67892))
        img = img.transpose(2, 0, 1)
        gt /= (gt.max() + 1e-8)
        gt = gt[np.newaxis, ...]
        skeleton /= (skeleton.max() + 1e-8)
        skeleton = skeleton[np.newaxis, ...]
        edge /= (edge.max() + 1e-8)
        edge = edge[np.newaxis, ...]
        return img, gt, skeleton, edge

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, config, shuffle=True, num_workers=12, pin_memory=True):

    dataset = SalObjDataset(image_root, gt_root, config.trainsize, config.is_agu)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config.batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image, name = load_image_test(self.images[self.index], self.testsize)
        gt = load_label(self.gts[self.index], self.testsize)
        image = torch.from_numpy(image)
        gt = torch.from_numpy(gt)
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


def load_image_test(path, size):
    im = cv2.imread(path)
    name = path.split('/')[-1].split('.jpg')[0]
    in_ = np.array(im, dtype=np.float32)
    in_ = cv2.resize(in_, (size, size))
    if len(in_.shape) == 2:
        img_temp = np.zeros((in_.shape[0], in_.shape[1], 3))
        img_temp[:, :, 0] = in_
        img_temp[:, :, 0] = in_
        img_temp[:, :, 0] = in_
        in_ = img_temp
    in_ -= np.array((104.00699, 116.66877, 122.67892))

    in_ = in_.transpose((2, 0, 1))
    in_ = in_[np.newaxis, ...]
    return in_, name


def load_label(path, size):
    gt = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    gt = np.float32(gt)
    gt /= (gt.max() + 1e-8)
    return gt
