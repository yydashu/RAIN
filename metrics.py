from sklearn import metrics
import numpy as np
import torch
import os
from  PIL import Image
import torch.nn as nn
import SimpleITK  as sitk
import matplotlib.pyplot as plt

class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos =x #torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w
        return -self.loss.sum()


def ACC(y_true,y_pred,th=0.5):
    """
    :param y_true:  5D=(batch,h,w,slice,1)
    :param y_pred: 5D=(batch,h,w,slice,1)
    :return:  Acc"""
    y_pred = (np.array(y_pred )> th ).astype(np.int).reshape(-1,1)
    y_true=(np.array(y_true)> th ).astype(np.int).reshape(-1,1)
    acc=np.sum(y_pred==y_true)/len(y_pred)
    return  acc

def fc_false_positive(y_true,y_pred,th=0.5): # 2D          # 存进来的是一个批次

    y_pred = (np.array(y_pred )> th).astype(np.int).reshape(-1,1)
    y_true=(np.array(y_true)> th ).astype(np.int).reshape(-1,1)    # (batch,1)
    negative=np.sum(1-y_true)
    if negative==0:
        return 0
    fp=np.sum((y_pred>y_true).astype(np.int))/negative
    return fp

def fc_false_negative(y_true,y_pred,th=0.5):  # 2D
    y_pred = (np.array(y_pred )> th ).astype(np.int).reshape(-1,1)
    y_true=(np.array(y_true)> th ).astype(np.int).reshape(-1,1)    # (batch,1)
    positive=np.sum(y_true)
    if positive==0:
        return 0
    fn=np.sum((y_pred<y_true).astype(np.int))/positive
    return fn

def AUC(y_true,y_pred):
    y_pred = np.array(y_pred).reshape(-1,1)
    y_true = np.array(y_true).reshape(-1, 1)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    auc=metrics.auc(fpr, tpr)
    return auc,fpr, tpr,thresholds

def yuedenindex(fprs,tprs,thresholds):
    fprs = np.array(fprs ).reshape(-1,)
    tprs = np.array(tprs).reshape(-1,)
    thresholds=np.array(thresholds).reshape(-1,)
    yueden=tprs- fprs
    index=np.argmax( yueden)
    point=(tprs[index],1-fprs[index])
    return tprs[index],1-fprs[index],thresholds[index]

def yuedenindex_m(fprs,tprs,thresholds):
    fprs = np.array(fprs ).reshape(-1,)
    tprs = np.array(tprs).reshape(-1,)
    thresholds=np.array(thresholds).reshape(-1,)
    yueden=tprs*(1-fprs) # max(灵敏度+特异度-1)
    index=np.argmax( yueden)
    point=(tprs[index],1-fprs[index])
    return tprs[index],1-fprs[index],thresholds[index] ## 对应的灵敏度和特异度


