import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# fx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)
# fy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).astype(np.float32)
# fx = np.reshape(fx, (1, 1, 3, 3))
# fy = np.reshape(fy, (1, 1, 3, 3))
# fx = Variable(torch.Tensor(fx)).cuda()
# fy = Variable(torch.Tensor(fy)).cuda()
# contour_th = 1.5


# def label_edge_prediction(label):
#     # convert label to edge
#     label = label.gt(0.5).float()
#     label = F.pad(label, (1, 1, 1, 1), mode='replicate')
#     label_fx = F.conv2d(label, fx)
#     label_fy = F.conv2d(label, fy)
#     label_grad = torch.sqrt(torch.mul(label_fx, label_fx) + torch.mul(label_fy, label_fy))
#     label_grad = torch.gt(label_grad, contour_th).float()
#
#     return label_grad
#
#
# def pred_edge_prediction(pred):
#     pred = F.pad(pred, (1, 1, 1, 1), mode='replicate')
#     pred_fx = F.conv2d(pred, fx)
#     pred_fy = F.conv2d(pred, fy)
#     pred_grad = (pred_fx*pred_fx + pred_fy*pred_fy).sqrt().tanh()
##     return pred_fx, pred_fy, pred_grad


laplace = np.array(([-1, -1, -1], [-1, 8, -1], [-1, -1, -1]), dtype=np.float32)
laplace = laplace[np.newaxis, np.newaxis, ...]
laplace = torch.Tensor(laplace).cuda()


def edge_prediction(map):
    edge = F.conv2d(map, laplace, padding=1)
    edge = F.relu(torch.tanh(edge))
    return edge


def WCE(pred, label, is_logits=True):
    # weighted CE
    assert pred.size() == label.size()
    pred = pred.view(pred.size()[0], -1)
    label = label.view(label.size()[0], -1)
    label = torch.ge(label, 0.5).float()
    if is_logits:
        seg_pred = torch.sigmoid(pred)
    else:
        seg_pred = pred
    num_labels_pos = torch.sum(label, 1, keepdim=True)
    num_labels_neg = torch.sum(1.0 - label, 1, keepdim=True)
    num_total = num_labels_pos + num_labels_neg

    num_pred_TP = torch.sum(torch.mul(seg_pred, label), 1, keepdim=True)
    num_pred_TN = torch.sum(torch.mul(1-seg_pred, 1-label), 1, keepdim=True)

    weight0 = torch.div(num_labels_neg, num_total) + 1 - torch.div(num_pred_TP, num_labels_pos + 1e-8)
    weight1 = torch.div(num_labels_pos, num_total) + 1 - torch.div(num_pred_TN, num_labels_neg + 1e-8)

    weight = torch.mul(weight0.expand_as(label), label) + torch.mul(weight1.expand_as(label), 1 - label)

    loss = F.binary_cross_entropy_with_logits(pred, label, weight)
    return loss


def SoftmaxCrossEntropyWithLogits(pred, label):
    loss = torch.sum(- label * F.log_softmax(pred, 0), 0)
    return loss.sum()


def DiceLoss(pred, label):
    assert pred.size() == label.size()
    n, c, h, w = pred.size()
    pred = pred.view(n, -1)
    label = label.view(n, -1)
    union = (pred*label).sum(1)
    inter = (pred*pred).sum(1) + (label*label).sum(1)
    if union.max().data.cpu().numpy() == 0:
        return 0
    else:
        return 1 - torch.mean(2*(union+1) / (inter + 1))


def IOULoss(pred, label):
    assert pred.size() == label.size()
    inter = (pred*label).sum()
    union = pred.sum() + label.sum() - inter
    return 1 - inter / (union+1e-8)


def BoundaryRecall(pred, label):
    assert pred.size() == label.size()
    return 1 - (pred*label).sum()/label.sum()


def hieLoss(pred, label):
    assert pred.size() == label.size()
    label = label.gt(0).float()
    n, c, h, w = pred.size()
    pred = pred.view(n, -1)
    label = label.view(n, -1)
    ce = F.binary_cross_entropy_with_logits(pred, label)
    pred = pred.sigmoid()
    mae = (pred - label).abs().mean(dim=1)
    p = (pred*label).sum(dim=1) / pred.sum(dim=1)
    r = (pred*label).sum(dim=1) / label.sum(dim=1)
    f = 1.3*p*r / (0.3*p + r + 1e-8)

    return ce - 0.1*(p + r + f - mae).mean()


def Smooth(pred, label, alpha=0.5):
    diff = (pred - label).abs()
    loss1 = 0.5*diff*diff
    loss2 = alpha*diff - 0.5*alpha*alpha

    index = diff.ge(alpha).float()
    loss = loss2*index + loss1*(1-index)
    return loss.mean()


def FocalLoss(pred, label, gamma=2):
    logp = F.logsigmoid(pred)
    logpt = label*logp + (1-label)*(logp-pred)
    pt = logpt.exp()
    loss = -1 * (1 - pt)**gamma * logpt
    return loss.mean()


class AFN_Edge_Loss(nn.Module):
    # edge loss in attentive feedback network
    def __init__(self):
        super(AFN_Edge_Loss, self).__init__()
        self.pooling = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.loss = nn.MSELoss()

    def forward(self, pred, gt):
        edges_pred = torch.abs(pred-self.pooling(pred))
        edges_gt = torch.abs(gt-self.pooling(gt))

        edges_pred = self.min_max_norm(edges_pred)
        edges_gt = self.min_max_norm(edges_gt)

        return self.loss(edges_pred, edges_gt)

    def min_max_norm(self, in_):
        max_ = in_.max(dim=2)[0].max(dim=2)[0].unsqueeze(2).unsqueeze(3)
        min_ = in_.min(dim=2)[0].min(dim=2)[0].unsqueeze(2).unsqueeze(3)
        return (in_ - min_) / (max_ - min_ + 1e-8)


class AffinityLoss(nn.Module):
    # affinity loss in Adaptive Affinity fields for Semantic Segmentation (ECCV 2018)
    def __init__(self, kld_margin=1, kernel_size=7, dilation=1):
        super(AffinityLoss, self).__init__()
        self.kld_margin = kld_margin
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=kernel_size//2, dilation=dilation)

    def forward(self, pred, target):
        pred = F.interpolate(pred, scale_factor=0.25, mode='bilinear', align_corners=True).sigmoid()
        target = F.interpolate(target, scale_factor=0.25, mode='bilinear', align_corners=True)

        n, c, h, w = pred.size()
        pred_neighbor = self.unfold(pred).reshape((n, c, -1, h, w)).unsqueeze(-1).transpose(2, 5).contiguous()
        target_neighbor = self.unfold(target).reshape((n, c, -1, h, w)).unsqueeze(-1).transpose(2, 5).contiguous()

        pred_neighbor = pred_neighbor.squeeze(2)
        neg_pred_neighbor = 1 - pred_neighbor

        target_neighbor = target_neighbor.squeeze(2)
        target = target.unsqueeze(-1).expand_as(target_neighbor)
        edge_index = torch.ne(target, target_neighbor)
        not_edge_index = torch.eq(target, target_neighbor)

        pred = pred.unsqueeze(-1).expand_as(pred_neighbor)
        neg_pred = 1 - pred

        dist = self.l2(pred_neighbor, pred)
        dist += self.l2(neg_pred_neighbor, neg_pred)
        # dist = pred_neighbor*torch.log(pred_neighbor/(pred+1e-8)+1e-8)
        # dist += neg_pred_neighbor*torch.log(neg_pred_neighbor/(neg_pred+1e-8)+1e-8)

        not_edge_loss = dist
        edge_loss = torch.max(torch.zeros_like(dist), self.kld_margin-dist)

        edge_loss = torch.masked_select(edge_loss, edge_index).mean()
        not_edge_loss = torch.masked_select(not_edge_loss, not_edge_index).mean()

        return edge_loss, not_edge_loss

    def kldiv(self, in1, in2):
        return in1*torch.log(in1/(in2+1e-8) + 1e-8)

    def l2(self, in1, in2):
        return (in1 - in2)**2


class SCE(nn.Module):
    # structured cros  entropy loss
    # smoothing prediction in a local neighbour
    def __init__(self, kernel_size=5, dilation=1,  sort=False):
        super(SCE, self).__init__()
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=kernel_size//2, dilation=dilation)
        self.l1 = nn.L1Loss(reduction='none')
        self.sort = sort
        self.kernel_size = kernel_size

    def forward(self, pred, label):
        n, c, h, w = pred.size()
        pred_neighbor = self.unfold(pred).reshape((n, c, -1, h, w)).unsqueeze(-1).transpose(2, 5).contiguous().squeeze()
        label_neighbor = self.unfold(label).reshape((n, c, -1, h, w)).unsqueeze(-1).transpose(2, 5).contiguous().squeeze()

        label = label.squeeze().unsqueeze(-1).expand_as(label_neighbor)
        pred = pred.squeeze().unsqueeze(-1).expand_as(pred_neighbor)
        same_index = torch.eq(label, label_neighbor).float()

        loss_neighbor = self.l1(pred, pred_neighbor)
        loss1 = loss_neighbor * same_index

        # loss1 = loss1.max(dim=-1)[0]
        # topk
        n, h, w, k = loss1.size()
        top_num = (h*w*k)//10
        loss1 = loss1.view(n, -1).topk(k=top_num, dim=1)[0].mean()

        return loss1


class SAL(nn.Module):
    # scale-adaptive loss
    def __init__(self, kernel_size=15, kernel_size1=15):
        super(SAL, self).__init__()
        self.kernel_size = kernel_size
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)

    def forward(self, pred, label):
        d_g = self.pool(label)
        d_p = self.pool(pred.sigmoid())
        weight = (1 - torch.log(d_g + 1e-8))*label + (1 - label) * (1 + d_p)
        # d_p = self.pool(pred.sigmoid())
        # weight += d_p * self.kernel_size * self.kernel_size * (1 - label)
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
        loss = loss * weight
        return loss.mean()


def bce2d_new(input, target):
    assert(input.size() == target.size())
    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float()
    # ing = ((torch.gt(target, 0) & torch.lt(target, 1))).float()

    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg

    alpha = num_neg / num_total
    beta = 1.1 * num_pos / num_total
    weights = alpha * pos + beta * neg

    return F.binary_cross_entropy_with_logits(input, target, weights)

