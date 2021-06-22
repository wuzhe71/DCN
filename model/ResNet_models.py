import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import torchvision.models as models
import numpy as np
import random
import pdb
from .ResNet import ResNet50, ResNet101, ResNet50_Dropout
from .vgg import VGG_BN


def min_max_norm(in_):
    max_ = in_.max(dim=2)[0].max(dim=2)[0].unsqueeze(2).unsqueeze(3)
    min_ = in_.min(dim=2)[0].min(dim=2)[0].unsqueeze(2).unsqueeze(3)
    return (in_ - min_) / (max_ - min_ + 1e-8)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(BasicConv2d, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False, groups=groups),
            nn.BatchNorm2d(out_planes),
            nn.ReLU()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv_bn(x)
        return x


class PPM(nn.Module):
    # pyramid pooling module
    def __init__(self, channel):
        super(PPM, self).__init__()
        self.scales = [1, 2, 4, 8]
        self.poolings = [nn.AdaptiveAvgPool2d((s, s)) for s in self.scales]
        self.convs = nn.ModuleList([BasicConv2d(channel, channel, kernel_size=3, padding=1)
                                    for i in range(len(self.scales))])

        self.cat = BasicConv2d((len(self.scales)+1)*channel, channel, 1)

    def forward(self, x):
        pool_x = []
        for i, pooling in enumerate(self.poolings):
            pool_x.append(self.convs[i](pooling(x)))

        inp_x = []
        for i in range(len(self.scales)):
            inp_x.append(F.interpolate(pool_x[i], size=x.size()[2:], mode='bilinear', align_corners=True))
        inp_x.append(x)
        return self.cat(torch.cat(inp_x, dim=1))


class ASPP(nn.Module):
    # Atrous Spatial Pyramid Pooling
    def __init__(self, channel):
        super(ASPP, self).__init__()
        rates = [1, 3, 5, 7, 9]
        self.convs = nn.ModuleList([BasicConv2d(channel, channel, kernel_size=3, dilation=i, padding=i)
                                    for i in rates])

    def forward(self, x):
        out = 0
        for conv in self.convs:
            out += conv(x)
        return out


class Reduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Reduction, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)


class conv_upsample(nn.Module):
    def __init__(self, channel):
        super(conv_upsample, self).__init__()
        self.conv = BasicConv2d(channel, channel, 3, padding=1)

    def forward(self, x, target):
        if x.size()[2:] != target.size()[2:]:
            x = F.interpolate(x, size=target.size()[2:], mode='bilinear', align_corners=True)
        x = self.conv(x)
        return x


class ConcatOutput(nn.Module):
    def __init__(self, channel):
        super(ConcatOutput, self).__init__()
        self.conv_upsample1 = conv_upsample(channel)
        self.conv_upsample2 = conv_upsample(channel)
        self.conv_upsample3 = conv_upsample(channel)

        self.conv_cat1 = BasicConv2d(2*channel, channel, 3, padding=1)
        self.conv_cat2 = BasicConv2d(2*channel, channel, 3, padding=1)
        self.conv_cat3 = BasicConv2d(2*channel, channel, 3, padding=1)

        self.output = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )

    def forward(self, x1, x2, x3, x4):
        x3 = torch.cat((x3, self.conv_upsample1(x4, x3)), 1)
        x3 = self.conv_cat1(x3)

        x2 = torch.cat((x2, self.conv_upsample2(x3, x2)), 1)
        x2 = self.conv_cat2(x2)

        x1 = torch.cat((x1, self.conv_upsample3(x2, x1)), 1)
        x1 = self.conv_cat3(x1)

        x = self.output(x1)
        return x


class features(nn.Module):
    def __init__(self, channel):
        super(features, self).__init__()
        self.conv1 = BasicConv2d(channel, channel, 1)
        self.conv2 = BasicConv2d(channel, channel, 1)
        self.conv3 = BasicConv2d(channel, channel, 1)
        self.conv4 = BasicConv2d(channel, channel, 1)

    def forward(self, x1, x2, x3, x4):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)

        return x1, x2, x3, x4


class MS(nn.Module):
    def __init__(self, channel):
        super(MS, self).__init__()
        self.weights = Parameter(torch.zeros(channel, channel, 3, 3))
        self.weights.data.normal_(std=0.01)
        self.bn = nn.BatchNorm2d(channel)

    def forward(self, x):
        x1 = self.bn(F.conv2d(x, self.weights, padding=1, dilation=1))
        x2 = self.bn(F.conv2d(x, self.weights, padding=3, dilation=3))
        x3 = self.bn(F.conv2d(x, self.weights, padding=5, dilation=5))
        return x+x1+x2+x3


class DeNet(nn.Module):
    # Three Cross streams for saliency, edge, skeleton
    def __init__(self, channel=32, use_dropout=True):
        super(DeNet, self).__init__()
        if use_dropout:
            self.resnet = ResNet50_Dropout()
        else:
            self.resnet = ResNet50()

        self.reduce1 = Reduction(256, channel)
        self.reduce2 = Reduction(512, channel)
        self.reduce3 = Reduction(1024, channel)
        self.reduce4 = Reduction(2048, channel)

        self.sal_features = features(channel)
        self.edg_features = features(channel)
        self.ske_features = features(channel)

        self.ccs = nn.ModuleList([nn.Sequential(
            BasicConv2d(3*channel, channel, kernel_size=3, padding=1),
            BasicConv2d(channel, channel, kernel_size=3, padding=1)
        ) for i in range(4)])
        self.cme = nn.ModuleList([nn.Sequential(
            BasicConv2d(3*channel, channel, kernel_size=3, padding=1),
            BasicConv2d(channel, channel, kernel_size=3, padding=1)
        ) for i in range(4)])
        self.cms = nn.ModuleList([nn.Sequential(
            BasicConv2d(3*channel, channel, kernel_size=3, padding=1),
            BasicConv2d(channel, channel, kernel_size=3, padding=1)
        ) for i in range(4)])

        self.conv_cats = nn.ModuleList([nn.Sequential(
            BasicConv2d(2*channel, channel, kernel_size=3, padding=1),
            BasicConv2d(channel, channel, kernel_size=3, padding=1)
        ) for i in range(9)])

        self.cus = nn.ModuleList([conv_upsample(channel) for i in range(9)])
        self.prediction = nn.ModuleList([nn.Sequential(
            BasicConv2d(channel, channel, kernel_size=3, padding=1),
            nn.Conv2d(channel, 1, 1)
        ) for i in range(3)])

        if self.training:
            self.initialize_weights()

    def forward(self, x):
        size = x.size()[2:]
        x1, x2, x3, x4 = self.resnet(x)

        x1 = self.reduce1(x1)
        x2 = self.reduce2(x2)
        x3 = self.reduce3(x3)
        x4 = self.reduce4(x4)
        xs = [x1, x2, x3, x4]
        x_sal1, x_sal2, x_sal3, x_sal4 = self.sal_features(x1, x2, x3, x4)
        x_edg1, x_edg2, x_edg3, x_edg4 = self.edg_features(x1, x2, x3, x4)
        x_ske1, x_ske2, x_ske3, x_ske4 = self.ske_features(x1, x2, x3, x4)

        x_sal4_n = self.ccs[0](torch.cat((x_sal4, x_edg4, x_ske4), 1)) + x_sal4
        x_edg4_n = self.cme[0](torch.cat((x_sal4, x_edg4, x_ske4), 1)) + x_edg4
        x_ske4_n = self.cms[0](torch.cat((x_sal4, x_edg4, x_ske4), 1)) + x_ske4

        x_sal3 = self.conv_cats[0](torch.cat((x_sal3, self.cus[0](x_sal4_n, x_sal3)), 1))
        x_edg3 = self.conv_cats[1](torch.cat((x_edg3, self.cus[1](x_edg4_n, x_edg3)), 1))
        x_ske3 = self.conv_cats[2](torch.cat((x_ske3, self.cus[2](x_ske4_n, x_ske3)), 1))

        x_sal3_n = self.ccs[1](torch.cat((x_sal3, x_edg3, x_ske3), 1)) + x_sal3
        x_edg3_n = self.cme[1](torch.cat((x_sal3, x_edg3, x_ske3), 1)) + x_edg3
        x_ske3_n = self.cms[1](torch.cat((x_sal3, x_edg3, x_ske3), 1)) + x_ske3

        x_sal2 = self.conv_cats[3](torch.cat((x_sal2, self.cus[3](x_sal3_n, x_sal2)), 1))
        x_edg2 = self.conv_cats[4](torch.cat((x_edg2, self.cus[4](x_edg3_n, x_edg2)), 1))
        x_ske2 = self.conv_cats[5](torch.cat((x_ske2, self.cus[5](x_ske3_n, x_ske2)), 1))

        x_sal2_n = self.ccs[2](torch.cat((x_sal2, x_edg2, x_ske2), 1)) + x_sal2
        x_edg2_n = self.cme[2](torch.cat((x_sal2, x_edg2, x_ske2), 1)) + x_edg2
        x_ske2_n = self.cms[2](torch.cat((x_sal2, x_edg2, x_ske2), 1)) + x_ske2

        x_sal1 = self.conv_cats[6](torch.cat((x_sal1, self.cus[6](x_sal2_n, x_sal1)), 1))
        x_edg1 = self.conv_cats[7](torch.cat((x_edg1, self.cus[7](x_edg2_n, x_edg1)), 1))
        x_ske1 = self.conv_cats[8](torch.cat((x_ske1, self.cus[8](x_ske2_n, x_ske1)), 1))

        x_sal1_n = self.ccs[3](torch.cat((x_sal1, x_edg1, x_ske1), 1)) + x_sal1
        x_edg1_n = self.cme[3](torch.cat((x_sal1, x_edg1, x_ske1), 1)) + x_edg1
        x_ske1_n = self.cms[3](torch.cat((x_sal1, x_edg1, x_ske1), 1)) + x_ske1

        pred_sal = self.prediction[0](x_sal1_n)
        pred_edg = self.prediction[1](x_edg1_n)
        pred_ske = self.prediction[2](x_ske1_n)

        pred_sal = F.interpolate(pred_sal, size=size, mode='bilinear', align_corners=True)
        pred_ske = F.interpolate(pred_ske, size=size, mode='bilinear', align_corners=True)
        pred_edg = F.interpolate(pred_edg, size=size, mode='bilinear', align_corners=True)

        return x_sal1_n, pred_sal, pred_edg, pred_ske

    def initialize_weights(self):
        pretrained_dict = torch.load('/userhome/code/pretrain/resnet50-19c8e357.pth')
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
        self.resnet.load_state_dict(all_params)


class EncoderDecoder(nn.Module):
    def __init__(self, channel):
        super(EncoderDecoder, self).__init__()
        self.conv1 = nn.Sequential(
            BasicConv2d(1, channel, 3, padding=1, stride=2),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv = nn.ModuleList([nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1, stride=2),
            BasicConv2d(channel, channel, 3, padding=1)
        ) for i in range(4)])
        self.conv_cat = nn.ModuleList([nn.Sequential(
            BasicConv2d(2*channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        ) for i in range(4)])

        self.conv_upsample = nn.ModuleList([conv_upsample(channel) for i in range(4)])

        self.prediction = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv[0](x1)
        x3 = self.conv[1](x2)
        x4 = self.conv[2](x3)
        x5 = self.conv[3](x4)

        if self.training:
            # for prediction
            batchsize, _, _, _ = x.size()
            split = int(batchsize/2)
            xs = [x1[:split, :, :, :], x2[:split, :, :, :], x3[:split, :, :, :], x4[:split, :, :, :], x5[:split, :, :, :]]
            # for gt
            x1 = x1[split:, :, :, :]
            x2 = x2[split:, :, :, :]
            x3 = x3[split:, :, :, :]
            x4 = x4[split:, :, :, :]
            x5 = x5[split:, :, :, :]

            x4 = self.conv_cat[0](torch.cat((x4, self.conv_upsample[0](x5, x4)), 1))
            x3 = self.conv_cat[1](torch.cat((x3, self.conv_upsample[1](x4, x3)), 1))
            x2 = self.conv_cat[2](torch.cat((x2, self.conv_upsample[2](x3, x2)), 1))
            x1 = self.conv_cat[3](torch.cat((x1, self.conv_upsample[3](x2, x1)), 1))

            return xs, self.prediction(x1)
        else:
            return [x1, x2, x3, x4, x5]


class Refine(nn.Module):
    def __init__(self, channel, use_rl):
        super(Refine, self).__init__()
        self.sal = EncoderDecoder(channel)
        self.use_rl = use_rl
        if use_rl:
            self.conv1x1 = BasicConv2d(128, channel, 1)
        self.conv1 = nn.Sequential(
            BasicConv2d(1, channel, 3, padding=1, stride=2),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv = nn.ModuleList([nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1, stride=2),
            BasicConv2d(channel, channel, 3, padding=1)
        ) for i in range(4)])
        self.conv_cat = nn.ModuleList([nn.Sequential(
            BasicConv2d(2*channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        ) for i in range(4)])
        self.conv_upsample = nn.ModuleList([conv_upsample(channel) for i in range(4)])
        # self.ppms = nn.ModuleList([PPM(channel) for i in range(3)])
        self.prediction = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )

    def forward(self, x, y, feature_x):
        # x for gt and sal
        # y for edge and ske
        if self.training:
            sal_features, gt_re = self.sal(x)
        else:
            sal_features = self.sal(x)
        x1, x2, x3, x4, x5 = sal_features
        # multi-scale features for y
        y1 = self.conv1(y)
        y2 = self.conv[0](y1)
        y3 = self.conv[1](y2)
        y4 = self.conv[2](y3)
        y5 = self.conv[3](y4)

        # completion
        c1 = x1 + y1
        c2 = x2 + y2
        c3 = x3 + y3
        c4 = x4 + y4
        c5 = x5 + y5

        # c5 = self.ppms[0](c5)
        c4 = self.conv_cat[0](torch.cat((c4, self.conv_upsample[0](c5, c4)), 1))
        # c4 = self.ppms[1](c4)
        c3 = self.conv_cat[1](torch.cat((c3, self.conv_upsample[1](c4, c3)), 1))
        # c3 = self.ppms[2](c3)
        c2 = self.conv_cat[2](torch.cat((c2, self.conv_upsample[2](c3, c2)), 1))
        c1 = self.conv_cat[3](torch.cat((c1, self.conv_upsample[3](c2, c1)), 1))

        feature_x = F.interpolate(feature_x, size=c1.size()[2:], mode='bilinear', align_corners=True)
        if self.use_rl:
            c1 = c1 + self.conv1x1(feature_x)
        if self.training:
            return self.prediction(c1), gt_re
        else:
            return self.prediction(c1)


class DCN(nn.Module):
    def __init__(self, channel):
        super(DCN, self).__init__()
        self.stage1 = Baseline1(channel)
        self.stage2 = Refine(channel//4, use_rl=False)

    def forward(self, ins):
        if self.training:
            x, gt = ins
        else:
            x = ins
        feature_sal, pred_sal, pred_edg, pred_ske = self.stage1(x)
        pred_ske = min_max_norm(pred_ske.sigmoid())
        if self.training:
            pred0, pred1 = self.stage2(torch.cat((pred_sal.sigmoid(), gt), 0), pred_edg.sigmoid() + pred_ske, feature_sal)
            pred0 = F.interpolate(pred0, size=x.size()[2:], mode='bilinear', align_corners=True)
            pred1 = F.interpolate(pred1, size=x.size()[2:], mode='bilinear', align_corners=True)
            return pred0, pred1
        else:
            return self.stage2(pred_sal.sigmoid(), pred_edg.sigmoid() + pred_ske, feature_sal)
