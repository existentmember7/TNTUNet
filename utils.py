import numpy as np
import cv2
import json
import glob
import os
import torch
import torch.nn as nn

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
    mask_path = path + 'label/'
    for file in glob.glob(img_path+'/*.png'):
        filename = file.split('/')[-1].split('.')[0]
        img_shape = cv2.imread(file).shape
        img = get_segmantation(img_shape, json_path+filename+'.json')
        cv2.imwrite(mask_path+filename+".png", img)

def get_depth(img_path, img_shape):
    img = cv2.resize(cv2.imread(img_path), img_shape, interpolation = cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,0]
    hsv = hsv.reshape(img_shape[0],img_shape[1], 1)
    return hsv

## end

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
## end