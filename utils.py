from cProfile import label
from fileinput import filename
from math import ceil
from tkinter import Label
import numpy as np
import cv2
import json
import glob
import tqdm
import os
import os.path as osp
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from einops import *
import csv
from matplotlib import pyplot as plt
import math

# <<data preprocessing>>
def get_filename(path):
    filename = path.split('\\')[-1]
    return filename

def generate_colors(class_num):
    RGB_list = []

    for i in range(class_num):
        R = np.random.randint(255)
        G = np.random.randint(255)
        B = np.random.randint(255)
        RGB_list.append([R,G,B])

    RGB_list = np.array(RGB_list)

    return RGB_list

def generate_colors(class_num):
    RGB_list = []

    for i in range(class_num):
        R = np.random.randint(255)
        G = np.random.randint(255)
        B = np.random.randint(255)
        RGB_list.append([R,G,B])

    RGB_list = np.array(RGB_list)

    return RGB_list

def crop2square(path):
    print("cropping image to squre of "+path+" ...")
    if osp.exists(osp.join(path, "crop")) == False:
        os.mkdir(osp.join(path, "crop"))
        os.mkdir(osp.join(osp.join(path, "crop"), "color"))
        os.mkdir(osp.join(osp.join(path, "crop"), "label"))
        os.mkdir(osp.join(osp.join(path, "crop"), "rgb"))
    else:
        return osp.join(path, "crop")
    ii_list = [0,0,1,1]
    rgb_list = generate_colors(34)
    label_count = []
    for filepath in tqdm(glob.glob(osp.join(osp.join(path, "color"),"*.png"))):
        filename = get_filename(filepath)
        img = cv2.imread(filepath)
        # print(osp.join(osp.join(path, "label"),filename))
        label = cv2.imread(osp.join(osp.join(path, "label"),filename), cv2.IMREAD_UNCHANGED)

        rgb_label = np.zeros((label.shape[0],label.shape[1],3))
        label_count += np.unique(label).tolist()
        for l in range(rgb_list.shape[0]):
            rgb_label[label == l, 0] = rgb_list[l,0]
            rgb_label[label == l, 1] = rgb_list[l,1]
            rgb_label[label == l, 2] = rgb_list[l,2]
        cv2.imwrite(osp.join(osp.join(osp.join(path, "crop"), "rgb"), filename), rgb_label)

        shape = img.shape
        ratio = shape[1]/shape[0]
        if ratio > 1.5:
            num = int(ratio)+1 if ratio > int(ratio) else int(ratio)
            # overlap = num * shape[0] - int(ratio)
            # overlap = int(overlap/num)
            for i in range(num):
                ii = ii_list[i]
                
                if (i+1)%2 == 1:
                    temp_img = img[:, ii*shape[0]:(ii+1)*shape[0]]
                    temp_label = label[:, ii*shape[0]:(ii+1)*shape[0]]
                    cv2.imwrite(osp.join(osp.join(osp.join(path, "crop"), "color"), str(i)+"_"+filename), temp_img)
                    cv2.imwrite(osp.join(osp.join(osp.join(path, "crop"), "label"), str(i)+"_"+filename), temp_label)
                else:
                    temp_img = img[:, -(ii+1)*shape[0]:shape[1]-ii*shape[0]]
                    temp_label = label[:, -(ii+1)*shape[0]:shape[1]-ii*shape[0]]
                    cv2.imwrite(osp.join(osp.join(osp.join(path, "crop"), "color"), str(i)+"_"+filename), temp_img)
                    cv2.imwrite(osp.join(osp.join(osp.join(path, "crop"), "label"), str(i)+"_"+filename), temp_label)
        else:
            temp_img = cv2.resize(img, (shape[0],shape[0]), interpolation = cv2.INTER_AREA)
            temp_label = cv2.resize(label, (shape[0],shape[0]), interpolation = cv2.INTER_AREA)
            cv2.imwrite(osp.join(osp.join(osp.join(path, "crop"), "color"), filename), temp_img)
            cv2.imwrite(osp.join(osp.join(osp.join(path, "crop"), "label"), filename), temp_label)
    print(np.unique(label_count))
    return osp.join(path, "crop")
# <<end>>


## for training loss
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        # print("target size: ", target.size())
        # target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target, softmax=False):
        # print(logit.shape, target.shape)
        if softmax:
            logit = torch.softmax(logit, dim=1)

        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]
        
        target_temp = []
        for i in range(len(target)):
            target_temp.append(decoding_label(target[i]))
        target = torch.from_numpy(np.array(target_temp))
        # print(logit.shape, target.shape)

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        # print(logit.shape, target.shape)
        # exit(-1)
        # 
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')
        
        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth/(num_class-1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
## end

def IoU(pred, target, n_classes, ignore_background):
# for mask and ground-truth label, not probability map
    ious = []
    iousSum = 0
    target = decoding_label(target)
    # pred = np.array(pred)
    # pred = torch.from_numpy(pred)
    pred = pred.view(-1)
    # target = np.array(target)
    target = torch.from_numpy(target)
    target = target.view(-1)

    ib = 0
    if ignore_background:
        ib = 1

    # Ignore IoU for background class ("0")
    for cls in range(ib, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
        union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(max(union, 1)))
            iousSum += float(intersection) / float(max(union, 1))
    return iousSum/(n_classes-ib)

def decoding_label(label):
    label = label.cpu()
    decoded_label = np.zeros((label.shape[1],label.shape[2]))
    for c in range(label.shape[0]):
        temp_label = label[c]
        decoded_label[temp_label == 1] = c
    return decoded_label

def encoding_label(labels, num_class):
    temp_labels_list = []
    for i in range(num_class):
        temp_labels_temp = np.zeros((labels.shape[0], labels.shape[1], 1))
        temp_labels_temp[labels == i] = 1
        temp_labels_list.append(temp_labels_temp)
    
    temp_labels = temp_labels_list[0]
    for i in range(1,len(temp_labels_list)):
        temp_labels = np.concatenate((temp_labels,temp_labels_list[i]), axis=2)

    return temp_labels

def learning_rate_policy(base_lr, num_iter, max_iter):
    lr = base_lr * (1.0 - num_iter/max_iter) ** 0.9
    return lr

## test only mIoU
def test_mIoU(test_result_folder_path, test_data_path):
    IoUs = []
    for file in glob.glob(test_result_folder_path+'*.png'):
        filename = file.split('/')[-1].split('.')[0]
        img_result = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        img_result[img_result>0] = 1
        img_result = np.reshape(img_result, (1, img_result.shape[0], img_result.shape[1]))
        img_data = cv2.imread(test_data_path+filename+'.png', cv2.IMREAD_UNCHANGED)
        img_result = torch.from_numpy(img_result)
        img_data = encoding_label(img_data, 2)
        img_data = rearrange(img_data, 'h w c -> c h w')
        img_data = torch.from_numpy(img_data)
        iou = IoU(img_result, img_data, 2, True)
        IoUs.append(iou)
    mean_IoU = np.mean(np.array(IoUs))
    print(mean_IoU)