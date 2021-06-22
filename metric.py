from __future__ import division

import numpy as np
from scipy import ndimage


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return np.mean(self.losses[np.maximum(len(self.losses)-self.num, 0):])


class cal_maxF(object):
    # max Fmeasure
    def __init__(self, num, thds=255):
        self.num = num
        self.thds = thds
        self.precision = np.zeros((self.num, self.thds))
        self.recall = np.zeros((self.num, self.thds))
        self.idx = 0

    def update(self, pred, gt):
        if gt.max() != 0:
            prediction, recall = self.cal(pred, gt)
            self.precision[self.idx, :] = prediction
            self.recall[self.idx, :] = recall
        self.idx += 1

    def cal(self, pred, gt):
        pred = np.uint8(pred*255)
        target = pred[gt > 0.5]
        nontarget = pred[gt <= 0.5]
        targetHist, _ = np.histogram(target, bins=range(256))
        nontargetHist, _ = np.histogram(nontarget, bins=range(256))
        targetHist = np.cumsum(np.flipud(targetHist), axis=0)
        nontargetHist = np.cumsum(np.flipud(nontargetHist), axis=0)
        precision = targetHist / (targetHist + nontargetHist + 1e-8)
        recall = targetHist / np.sum(gt)

        return precision, recall

    def show(self):
        assert self.num == self.idx
        precision = self.precision.mean(axis=0)
        recall = self.recall.mean(axis=0)
        fmeasure = 1.3 * precision * recall / (0.3 * precision + recall + 1e-8)

        return fmeasure.max()


class cal_mae(object):
    # mean absolute error
    def __init__(self):
        self.prediction = []

    def update(self, pred, gt):
        score = self.cal(pred, gt)
        self.prediction.append(score)

    def cal(self, pred, gt):
        gt[gt >= 0.5] = 1
        gt[gt < 0.5] = 0
        return np.mean(np.abs(pred-gt))

    def show(self):
        return np.mean(self.prediction)


class cal_meanF(object):
    # Fmeasure
    def __init__(self):
        self.prediction = []

    def update(self, pred, gt):
        score = self.cal(pred, gt)
        self.prediction.append(score)

    def cal(self, pred, gt):
        thd_gt = np.zeros(gt.shape)
        thd_pred = np.zeros(pred.shape)
        thd_gt[gt >= 0.5] = 1
        thd = 2 * np.mean(pred)
        if thd > 1:
            thd = 0.99
        thd_pred[pred >= thd] = 1
        p = np.sum(thd_pred * thd_gt) / (np.sum(thd_pred) + 1e-8)
        r = np.sum(thd_pred * thd_gt) / (np.sum(thd_gt) + 1e-8)

        return (1 + 0.3) * p * r / (0.3 * p + r + 1e-8)

    def show(self):
        return np.mean(self.prediction)


class cal_acc(object):
    # accuracy
    def __init__(self):
        self.prediction = []

    def update(self, pred, gt):
        score = self.cal(pred, gt)
        self.prediction.append(score)

    def cal(self, pred, gt):
        return np.sum(pred*gt+(1-pred)*(1-gt)) / gt.size

    def show(self):
        return np.mean(self.prediction)


class cal_ber(object):
    #balance error rate on set
    def __init__(self, size):
        self.size = size
        self.score = np.zeros((size, 4))
        self.idx = 0

    def update(self, pred, gt):
        self.score[self.idx, :] = self.cal(pred, gt)
        self.idx += 1

    def cal(self, pred, gt):
        posPoints = gt > 0.5
        negPoints = gt <= 0.5
        countPos = np.sum(posPoints)
        countNeg = np.sum(negPoints)

        posPred = pred > 0.5
        negPred = pred <= 0.5
        tp = posPred * posPoints
        tn = negPred * negPoints
        countTP = np.sum(tp)
        countTN = np.sum(tn)

        return np.array([countTP, countTN, countPos, countNeg])

    def show(self):
        assert self.idx == self.size
        posAcc = np.sum(self.score[:, 0]) / np.sum(self.score[:, 2])
        negAcc = np.sum(self.score[:, 1]) / np.sum(self.score[:, 3])
        BER = 0.5 * (2 - posAcc - negAcc)
        return 100*BER


class cal_sm(object):
    # structure similarity
    def __init__(self, alpha=0.5):
        self.prediction = []
        self.alpha = alpha

    def update(self, pred, gt):
        gt = gt > 0.5
        score = self.cal(pred, gt)
        self.prediction.append(score)

    def show(self):
        return np.mean(self.prediction)

    def cal(self, pred, gt):
        y = np.mean(gt)
        if y == 0:
            score = 1 - np.mean(pred)
        elif y == 1:
            score = np.mean(pred)
        else:
            score = self.alpha*self.object(pred, gt) + (1 - self.alpha) * self.region(pred, gt)
        return score

    def object(self, pred, gt):
        fg = pred*gt
        bg = (1-pred)*(1 - gt)

        u = np.mean(gt)
        return u*self.s_object(fg, gt) + (1-u)*self.s_object(bg, np.logical_not(gt))

    def s_object(self, in1, in2):
        x = np.mean(in1[in2])
        sigma_x = np.std(in1[in2])
        return 2*x / (pow(x, 2) + 1 + sigma_x + 1e-8)

    def region(self, pred, gt):
        [y, x] = ndimage.center_of_mass(gt)
        y = int(round(y)) + 1
        x = int(round(x)) + 1
        [gt1, gt2, gt3, gt4, w1, w2, w3, w4] = self.divideGT(gt, x, y)
        pred1, pred2, pred3, pred4 = self.dividePred(pred, x, y)

        score1 = self.ssim(pred1, gt1)
        score2 = self.ssim(pred2, gt2)
        score3 = self.ssim(pred3, gt3)
        score4 = self.ssim(pred4, gt4)

        return w1*score1 + w2*score2 + w3*score3 + w4*score4

    def divideGT(self, gt, x, y):
        h, w = gt.shape
        area = h*w
        LT = gt[0:y, 0:x]
        RT = gt[0:y, x:w]
        LB = gt[y:h, 0:x]
        RB = gt[y:h, x:w]

        w1 = x*y / area
        w2 = y*(w-x) / area
        w3 = (h-y)*x / area
        w4 = (h-y)*(w-x) / area

        return LT, RT, LB, RB, w1, w2, w3, w4

    def dividePred(self, pred, x, y):
        h, w = pred.shape
        LT = pred[0:y, 0:x]
        RT = pred[0:y, x:w]
        LB = pred[y:h, 0:x]
        RB = pred[y:h, x:w]

        return LT, RT, LB, RB

    def ssim(self, in1, in2):
        in2 = np.float32(in2)
        h, w = in1.shape
        N = h*w

        x = np.mean(in1)
        y = np.mean(in2)
        sigma_x = np.var(in1)
        sigma_y = np.var(in2)
        sigma_xy = np.sum((in1-x)*(in2-y)) / (N-1)

        alpha = 4 * x * y * sigma_xy
        beta = (x*x + y*y)*(sigma_x + sigma_y)

        if alpha != 0:
            score = alpha / (beta + 1e-8)
        elif alpha == 0 and beta == 0:
            score = 1
        else:
            score = 0

        return score