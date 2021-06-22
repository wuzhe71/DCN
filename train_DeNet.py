from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch import optim
import torchvision.transforms as transforms

import numpy as np
import cv2, pdb, os, sys, argparse
from datetime import datetime
from scipy import misc
sys.path.append('..')

from data import get_loader, test_dataset
from metric import AvgMeter, cal_mae, cal_maxF, cal_sm
from loss import edge_prediction, AFN_Edge_Loss, AffinityLoss, SCE, SAL
from model.ResNet_models import DeNet

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=50, help='epoch number')
parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
parser.add_argument('--batchsize', type=int, default=12, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--trainset', type=str, default='DUTS_TRAIN', help='training  dataset')
parser.add_argument('--channel', type=int, default=128, help='channel number of convolutional layers in decoder')
parser.add_argument('--iter_num', type=int, default=1, help='iteration number')
parser.add_argument('--is_ms', type=bool, default=True, help='multi scale training')
parser.add_argument('--is_agu', type=bool, default=False, help='using data augmentation')

config = parser.parse_args()
print(config)
if config.is_ms:
    size_rates = [0.75, 1, 1.25]
    config.lr = 0.002
else:
    size_rates = [1]
    config.lr = 0.005

np.random.seed(2019)
torch.manual_seed(2019)
torch.cuda.manual_seed(2019)

data_path = '/userhome/data/SOD/'
image_root = data_path + config.trainset + '/images/'
gt_root = data_path + config.trainset + '/gts/'
train_loader = get_loader(image_root, gt_root, config)
total_step = len(train_loader)
CE = torch.nn.BCEWithLogitsLoss()
CE1 = torch.nn.BCELoss()
edge_loss = AFN_Edge_Loss()
structure_loss = AffinityLoss()
sal = SAL()
sce = SCE()


def train(train_loader, model, optimizer, epoch, size_rates):
    model.train()
    loss_record, loss_record1, loss_record2 = AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # Load data
            images, gts, skeletons, edges = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            skeletons = Variable(skeletons).cuda()
            # edges = Variable(edges).cuda()
            edges = edge_prediction(gts)
            trainsize = int(round(config.trainsize*rate/32)*32)
            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                skeletons = F.interpolate(skeletons, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                edges = F.interpolate(edges, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            _, pred_sal, pred_edg, pred_ske = model(images)

            loss1 = CE(pred_sal, gts)
            loss2 = CE(pred_edg, edges)
            loss3 = CE(pred_ske, skeletons)
            loss = loss1 + loss2 + loss3
            loss.backward()

            optimizer.step()
            if rate == 1:
                loss_record.update(loss1.data.cpu().numpy(), config.batchsize)
                loss_record1.update(loss2.data.cpu().numpy(), config.batchsize)
                loss_record2.update(loss3.data.cpu().numpy(), config.batchsize)

        if i % 1500 == 0 or i == total_step:
            log = '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss_sal: {:.4f} Loss_ske: {:.4f} Loss_edg: {:.4f}'.\
                format(datetime.now(), epoch, config.epoch, i, total_step,
                       loss_record.show(), loss_record1.show(), loss_record2.show())

            print(log)

print("Let's go!")
model = DeNet(channel=config.channel, use_dropout=True)
model.cuda()

params = model.parameters()
optimizer = optim.SGD(params, lr=config.lr, momentum=0.9, weight_decay=5e-4, nesterov=False)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epoch)
for epoch in range(0, config.epoch):
    scheduler.step()
    train(train_loader, model, optimizer, epoch, size_rates)
torch.save(model.state_dict(), 'DeNet.pth')

