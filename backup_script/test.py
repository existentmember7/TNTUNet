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
from sklearn.metrics import classification_report, jaccard_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import glob
import copy
from utils import *

# for icshm
# def save_testing_result(outputs, ds, i):
#     output = np.array(outputs[0].cpu())
#     output =  np.reshape(output, (output.shape[1], output.shape[2], output.shape[0]))
#     img = np.zeros((output.shape[0], output.shape[1], 3))
#     # class_color = [[70,70,70],[150,150,202],[100,186,198],[186,183,167],[133,255,255],[206,192,192],[160,80,32],[1,134,193],[0,0,255],[203,192,255],[50,225,255]]
#     # class_color = [[70,70,70],[0,0,255],[203,192,255],[50,225,255]]
#     # class_color = [[70,70,70],[50,255,255]]
#     class_color = [[70,70,70],[150,150,202],[100,186,198],[186,183,167],[133,255,255],[206,192,192],[160,80,32],[1,134,193]]
#     for c in  range(len(class_color)):
#         for w in range(output.shape[0]):
#             for h in range(output.shape[1]):
#                 #print(output.shape)
#                 #print(output[w][h])
#                 #exit(-1)
#                 if output[w][h][0] == c:
#                     img[w][h] = class_color[c]
#                 # img[output==c] = class_color[c]
#     # outputs = np.array(outputs[0].cpu())
#     # outputs[outputs!=1] = 255
#     # cv2.imwrite("/home/user/Documents/han/TNTUNet/test_result_img/"+ds.filenames[i]+'.png', np.reshape(outputs, (outputs.shape[1], outputs.shape[2], outputs.shape[0])))
#     cv2.imwrite("/home/wisccitl/Documents/han/TNTUNet/test_result_img/"+ds.filenames[i]+'.png', img)

# def save_testing_result(outputs, ds, i):
#     output = np.array(outputs[0].cpu())
#     output =  np.reshape(output, (output.shape[1], output.shape[2]))

#     temp_index_1 = output>255
#     temp_index_2 = output<=255
#     img =np.zeros((output.shape[0], output.shape[1], 3))
#     img[:,:,0][temp_index_1] = 255
#     img[:,:,1][temp_index_1] = output[temp_index_1]%255
#     img[:,:,0][temp_index_2] = output[temp_index_2]

#     cv2.imwrite("/home/wisccitl/Documents/han/TNTUNet/test_result_img/"+ds.filenames[i]+'.png', img)

def show_index(test_result, test_label):
    y_pred = test_result.flatten()
    y_true = test_label.flatten()
    
    # target_names = ['ignore','wall','beam','column','window frame','window pane','balcony','slab','crack','spall','rebar']
    # target_names = ['ignore','crack','spell','rebar']
    # target_names = ['ignore','slab']
    # target_names = ['wall','beam','column','window frame','window pane','balcony','slab']

    # for itann term project
    target_names = []
    for i in range(11):
        target_names.append(str(i))
    

    # y_pred[y_pred == 0] = 3
    print(classification_report(y_true, y_pred, target_names=target_names))
    print ("**************************************************************")

    plt.figure()
    cnf_matrix = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cnf_matrix, classes=target_names,normalize=True, title="confusion matrix")

    # plt.show()
    plt.savefig("matrix.png")

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    fig = plt.figure(figsize=(10,10), dpi=800)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, format(cm[i, j], fmt),
    #              horizontalalignment="center",
    #              color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def inference(model, testing_data, ignore_background, n_classes):
    ds = testing_data.dataset
    model.eval()
    # mIoUs = []

    result = []
    label = []
    mask = []
    
    device = torch.device("cuda:0")

    for i, (i_batch, i_label) in tqdm(enumerate(testing_data)):
        i_batch, i_label = i_batch.cuda().to(device), i_label.type(torch.FloatTensor).cuda().to(device)
        outputs = model(i_batch)
        outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
        # save_testing_result(outputs, ds, i)
        # for j in range(len(outputs)):
        #     mIoU = IoU(outputs[j], i_label[j], n_classes, ignore_background)
        #     mIoUs.append(mIoU)
    
        if len(result) == 0:
            result = outputs.cpu().numpy()
            label = torch.argmax(i_label,dim=1,keepdim=True).cpu().numpy()
        else:
            result = np.concatenate((result, outputs.cpu().numpy()))
            label = np.concatenate((label, torch.argmax(i_label,dim=1,keepdim=True).cpu().numpy()))
        
       # if len(result)>=10:
       #     break

    # mean_IoU = np.mean(np.array(mIoUs))
    
    result = result.reshape(result.shape[0]*result.shape[1]*result.shape[2]*result.shape[3])
    label = label.reshape(label.shape[0]*label.shape[1]*label.shape[2]*label.shape[3])
    # result = result[label!=0]
    # label = label[label!=0]
    
    print("result:",result)
    print("label:",label)

    ious = jaccard_score(label, result, average='weighted')

    show_index(result, label)

    return ious
        
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

    model = TNTUNet(image_size=opt.image_width, class_num=opt.num_classes, channels=opt.channels).cuda()
    model.load_state_dict(torch.load(opt.model_weight_path,map_location='cuda:0'))
    
    device = torch.device("cuda:0")
    model.to(device)

    mean_IoU = inference(model, testing_data, opt.ignore_background_class, opt.num_classes)

    print("mean_IoU: ", mean_IoU)
    
    #filenames = glob.glob("test_result_image")
    #file_dict = {}
    #count = 0
    #while True:
        #np.zeros((1920,1080))




