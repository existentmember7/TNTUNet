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
from datetime import datetime

def show_index(test_result, test_label, opt):
    y_pred = test_result.flatten()
    y_true = test_label.flatten()

    # for itann term project
    # target_names = []
    # for i in range(11):
    #     target_names.append(str(i))
    
    # for CS766
    _mask_labels = {0: 'Sky', 1: 'Building', 2: 'Road', 3: 'Sidewalk',
                4: 'Fence', 5: 'grass', 6: 'Pole', 7: 'Car',
                8: 'Sign', 9: 'People', 10: 'Cyclist', 11: 'void'}
    # class_list = [0,1,2,3,5,7]
    class_list = [4,6,8,9,10,11]
    target_names = []
    for key in class_list:
        target_names.append(_mask_labels[key])
    # for key in _mask_labels:
    #     target_names.append(_mask_labels[key])
    # target_names.pop()
    
    print(classification_report(y_true, y_pred, target_names=target_names))
    print ("**************************************************************")

    plt.figure()
    cnf_matrix = confusion_matrix(y_true, y_pred)

    plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True, title="confusion matrix")

    # plt.show()
    plt.savefig(osp.join(opt.results_save_path,"matrix.png"))

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
    cm = np.rint(np.array(cm)*100)

    # plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # # fig = plt.figure(figsize=(10,10), dpi=800)
    # plt.title(title)
    # plt.colorbar()
    # tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45)
    # plt.yticks(tick_marks, classes)

    # fmt = '.2f' if normalize else 'd'
    # thresh = cm.max() / 2.

    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # plt.tight_layout()
    import seaborn as sns
    ax = sns.heatmap(cm, annot=True, cmap='Blues', annot_kws={"size":8})

    ax.set_title(title)
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values ')

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(classes, rotation=30)
    ax.yaxis.set_ticklabels(classes, rotation=30)

def inference(model, testing_data, opt):

    now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    os.mkdir(osp.join(opt.results_save_path, now))
    opt.results_save_path = osp.join(opt.results_save_path, now)

    ds = testing_data.dataset
    model.eval()

    result = []
    label = []

    RGB_list = generate_colors(opt.num_classes)
    
    device = torch.device("cuda:0")

    for i, (i_batch, i_label) in tqdm(enumerate(testing_data)):
        i_batch, i_label = i_batch.cuda().to(device), i_label.type(torch.FloatTensor).cuda().to(device)
        outputs = model(i_batch)
        outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
        save_image_results(outputs, ds, i, RGB_list)
    
        if len(result) == 0:
            result = outputs.cpu().numpy()
            label = torch.argmax(i_label,dim=1,keepdim=True).cpu().numpy()
        else:
            result = np.concatenate((result, outputs.cpu().numpy()))
            label = np.concatenate((label, torch.argmax(i_label,dim=1,keepdim=True).cpu().numpy()))
    
    result = result.reshape(result.shape[0]*result.shape[1]*result.shape[2]*result.shape[3])
    label = label.reshape(label.shape[0]*label.shape[1]*label.shape[2]*label.shape[3])
    
    print("result:",result)
    print("label:",label)

    ious = jaccard_score(label, result, average='weighted')

    combine_image(opt)
    
    show_index(result, label, opt)

    return ious
      
def decoding_label(label):
    label = label.cpu()
    decoded_label = np.zeros((label.shape[1],label.shape[2]))
    for c in range(label.shape[0]):
        temp_label = label[c]
        decoded_label[temp_label == 1] = c
    return decoded_label

def save_image_results(output, ds, i,class_color):
    c,c,w,h = output.shape
    img = np.zeros((w,h,3))
    output = output[0,...].clone().cpu()
    output = output.reshape((h, w))
    for j in range(class_color.shape[0]):
        img[output == j, 0] = class_color[j,0]
        img[output == j, 1] = class_color[j,1]
        img[output == j, 2] = class_color[j,2]
    
    cv2.imwrite(osp.join(opt.results_save_path, ds.filenames[i]+'.png'), img)

def generate_colors(class_num):
    RGB_list = []

    for i in range(class_num):
        R = np.random.randint(255)
        G = np.random.randint(255)
        B = np.random.randint(255)
        RGB_list.append([R,G,B])

    RGB_list = np.array(RGB_list)

    return RGB_list

def combine_image(opt):
    filenames = glob.glob(osp.join(opt.results_save_path, "*.png"))
    file_dict = []
    for count in tqdm(range(len(filenames))):
        filename = filenames[count].split('\\')[-1].split('_')[-1]
        # print(filename)
        if filename not in file_dict:
            img = np.zeros((376,1241,3))
            fileset = glob.glob(osp.join(opt.results_save_path, "*_"+filename))
            fileset = sorted(fileset)
            img_1 = cv2.imread(fileset[0])
            img_2 = cv2.imread(fileset[1])
            img_3 = cv2.imread(fileset[2])
            img_4 = cv2.imread(fileset[3])
            img_1 = cv2.resize(img_1, (376, 376), interpolation = cv2.INTER_AREA)
            img_2 = cv2.resize(img_2,(376, 376), interpolation = cv2.INTER_AREA)
            img_3 = cv2.resize(img_3, (376, 376), interpolation = cv2.INTER_AREA)
            img_4 = cv2.resize(img_4,(376, 376), interpolation = cv2.INTER_AREA)
            img[:,:376] = img_1
            img[:,376:376*2] = img_3
            img[:,-376*2:-376] = img_4
            img[:,-376:] = img_2
            cv2.imwrite(osp.join(opt.results_save_path, filename+".png"), img)
        
        file_dict.append(filename)
        os.remove(filenames[count])

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

    opt.testing_data_path = crop2square(opt.testing_data_path)
    testing_data = DataLoader(CustomDataset(opt), batch_size=opt.batch_size, shuffle=False)

    model = TNTUNet(image_size=opt.image_width, class_num=opt.num_classes, channels=opt.channels).cuda()
    model.load_state_dict(torch.load(opt.model_weight_path,map_location='cuda:0'))
    
    device = torch.device("cuda:0")
    model.to(device)

    mean_IoU = inference(model, testing_data, opt)

    print("mean_IoU: ", mean_IoU)




