import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.datasets import CustomDataset
from option.option import Option
from networks.TNTUNet import TNTUNet
import cv2
# from utils import *

def save_testing_result(outputs, ds, i):
    outputs = np.array(outputs[0].cpu())
    outputs[outputs!=1] = 255
    cv2.imwrite("/home/Documents/han/TNTUNet/test_result_img/"+ds.filenames[i]+'.png', np.reshape(outputs, (outputs.shape[1], outputs.shape[2], outputs.shape[0])))

def inference(model, testing_data, ignore_background, n_classes):
    ds = testing_data.dataset
    model.eval()
    mIoUs = []
    for i, (i_batch, i_label) in tqdm(enumerate(testing_data)):
        i_batch, i_label = i_batch.cuda(), i_label.type(torch.FloatTensor).cuda()
        outputs = model(i_batch)
        outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
        save_testing_result(outputs, ds, i)
        for i in range(len(outputs)):
            mIoU = IoU(outputs[i], i_label[i], n_classes, ignore_background)
            mIoUs.append(mIoU)
    mean_IoU = np.mean(np.array(mIoUs))
    return mean_IoU
        
def decoding_label(label):
    label = label.cpu()
    decoded_label = np.zeros((label.shape[1],label.shape[2]))
    for c in range(label.shape[0]):
        temp_label = label[c]
        decoded_label[temp_label == 1] = c
    return decoded_label


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

    


if __name__ == "__main__":
    opt = Option().opt

    ## training random seed control
    if not opt.deterministic:
        cudnn.deterministic = False
        cudnn.benchmark = True
    else:
        cudnn.deterministic = True
        cudnn.benchmark = False
    
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    testing_data = DataLoader(CustomDataset(opt), batch_size=opt.batch_size, shuffle=False)

    model = TNTUNet(image_size=opt.image_width, class_num=opt.num_classes).cuda()
    model.load_state_dict(torch.load(opt.model_weight_path))

    mean_IoU = inference(model, testing_data, opt.ignore_background_class, opt.num_classes)

    print("mean_IoU: ", mean_IoU)
    
