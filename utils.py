import numpy as np
import cv2
import json
import glob
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

## for preparing datasets
def get_segmantation(img_shape, label_path):
    spacing_label = get_spacing_label(label_path)
    img = get_masks(img_shape, spacing_label, 255)
    return img

def get_spacing_label(label_path):
    with open(label_path) as f:
        label = json.load(f)
        keyValList = ['spacing']
        spacing_labels = list(filter(lambda d: d['label'] in keyValList, label['shapes']))
        return spacing_labels

def get_masks(img_shape, labels, class_No):
    img = np.zeros((img_shape[0], img_shape[1], 1), dtype=np.uint8)
    for l in labels:
        points = np.array(l['points'], dtype=int)
        img = cv2.line(img, (points[0][0],points[0][1]), (points[1][0],points[1][1]), class_No, 10)
    return img

def generate_masks(path):
    img_path = path + 'color/'
    json_path = path + 'json/'
    mask_path = path + 'mask/'
    for file in glob.glob(img_path+'/*.png'):
        filename = file.split('/')[-1].split('.')[0]
        img_shape = cv2.imread(file).shape
        img = get_segmantation(img_shape, json_path+filename+'.json')
        cv2.imwrite(mask_path+filename+".png", img)

def mask2label(path, label_number):
    for img_path in glob.glob(path+'mask/*.png'):
        filename = img_path.split('/')[-1].split('.')[0]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img[img > 0] = label_number
        cv2.imwrite(path+"label/"+filename+".png", img)

def crop2split_data(path):
    folders = ['color', 'depth', 'label', 'mask']
    for folder in folders:
        for file in glob.glob(path+folder+'/*.png'):
            crop_and_split(file, (720, 720), path[:-1]+"_v2/"+folder+"/")

def crop_and_split(img_path, size, save_path):
    filename = img_path.split('/')[-1].split('.')[0]
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img1 = img[:size[0], :size[1]]
    img2 = img[:size[0], img.shape[1]-size[1]:]
    cv2.imwrite(save_path+filename+'_l.png', img1)
    cv2.imwrite(save_path+filename+'_r.png', img2)

def crop2split_data_3(path):
    folders = ['color', 'depth', 'label', 'mask']
    for folder in folders:
        for file in glob.glob(path+folder+'/*.png'):
            crop_and_split_3(file, (360, 360), path[:-4]+"_v3/"+folder+"/")

def crop_and_split_3(img_path, size, save_path):
    filename = img_path.split('/')[-1].split('.')[0]
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img1 = img[:size[0], :size[1]]
    img2 = img[:size[0], size[1]:]
    img3 = img[size[0]:, :size[1]] 
    img4 = img[size[0]:, size[1]:]
    cv2.imwrite(save_path+filename+'_1.png', img1)
    cv2.imwrite(save_path+filename+'_2.png', img2)
    cv2.imwrite(save_path+filename+'_3.png', img3)
    cv2.imwrite(save_path+filename+'_4.png', img4)



def get_depth(img_path, img_shape):
    img = cv2.resize(cv2.imread(img_path), img_shape, interpolation = cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,0]
    hsv = hsv.reshape(img_shape[0],img_shape[1], 1)
    return hsv

## end

# generate_masks('/media/han/D/aicenter_rebar_data/data/validation/')
# mask2label('/media/han/D/aicenter_rebar_data/data/validation/', 1)
crop2split_data_3("/media/han/D/aicenter_rebar_data/data/validation_v2/")

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

def learning_rate_policy(base_lr, num_iter, max_iter):
    lr = base_lr * (1.0 - num_iter/max_iter) ** 0.9
    return lr
