from __future__ import division
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch import optim
import torchvision.transforms as transforms


import numpy as np
import os, sys, argparse
from datetime import datetime
from scipy import misc
from PIL import Image
sys.path.append('..')

from data import get_loader, test_dataset
from metric import AvgMeter, cal_mae, cal_maxF, cal_sm
from model.ResNet_models import DCN

parser = argparse.ArgumentParser()
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--channel', type=int, default=128, help='channel number of convolutional layers in decoder')

config = parser.parse_args()
print(config)

np.random.seed(2019)
torch.manual_seed(2019)
torch.cuda.manual_seed(2019)

data_path = 'D:/code/SalientObject/dataset/'

model = DCN(channel=config.channel)
if torch.cuda.is_available():
    model.cuda()
model.load_state_dict(torch.load('./models/DCN.pth', map_location='cpu'))

valset = ['ECSSD', 'HKUIS', 'PASCAL', 'DUT-OMRON', 'DUTS-TEST']
model.eval()
for dataset in valset:
    save_path = './saliency_maps/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_root = data_path + dataset + '/images/'
    gt_root = data_path + dataset + '/gts/'
    test_loader = test_dataset(image_root, gt_root, config.trainsize)
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        file = save_path + name + '.png'
        gt = np.array(gt).astype('float')
        gt = gt / (gt.max() + 1e-8)
        if torch.cuda.is_available():
            image = Variable(image).cuda()
        else:
            image = Variable(image)

        res = model(image)
        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=True)
        res = res.sigmoid().data.cpu().numpy().squeeze()

        res = Image.fromarray(np.uint8(255*res)).convert('RGB')
        res.save(save_path+name+'.png')

