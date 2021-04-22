from numpy.lib import utils
from torch.utils.data import Dataset
import cv2
import glob
import os
from utils import *
import numpy as np
from einops import *

class CustomDataset(Dataset):

    def __init__(self, _opt, val=False):
        self.opt = _opt
        self.data_path = ""
        if val == True:
            self.data_path = self.opt.validating_data_path
        else:
            if self.opt.train:
                self.data_path = self.opt.training_data_path
            elif self.opt.test:
                self.data_path = self.opt.testing_data_path
        self.filenames = [os.path.basename(f).split('.')[0] for f in glob.glob(self.data_path+"color/*.png")]

    def encoding_label(self, labels):
        temp_labels_list = []
        for i in range(self.opt.num_classes):
            temp_labels_temp = np.zeros((labels.shape[0], labels.shape[1], 1))
            temp_labels_temp[labels == i] = 1
            temp_labels_list.append(temp_labels_temp)
        
        temp_labels = temp_labels_list[0]
        for i in range(1,len(temp_labels_list)):
            temp_labels = np.concatenate((temp_labels,temp_labels_list[i]), axis=2)

        return temp_labels


    def __getitem__(self, idx):
        depth = get_depth(self.data_path+'depth/'+self.filenames[idx]+".png", (self.opt.image_width, self.opt.image_height))
        color = cv2.resize(cv2.imread(self.data_path+'color/'+self.filenames[idx]+".png"), (self.opt.image_width, self.opt.image_height), interpolation = cv2.INTER_AREA)
        label = cv2.resize(cv2.imread(self.data_path+'label/'+self.filenames[idx]+".png", cv2.IMREAD_UNCHANGED), (self.opt.image_width, self.opt.image_height), interpolation = cv2.INTER_AREA)
        img = np.concatenate((color, depth), axis=2)
        img = rearrange(img, 'h w c -> c h w')
        label = self.encoding_label(label)
        label = rearrange(label, 'h w c -> c h w')

        return img.astype(np.float32), label.astype(np.float32)

    def __len__(self):
        return len(self.filenames)
