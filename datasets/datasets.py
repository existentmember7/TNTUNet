from numpy.lib import utils
from torch.utils.data import Dataset
import cv2
import glob
import os
from backup_script.utils import *
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
        self.filenames = [os.path.basename(f).split('.')[0] for f in glob.glob(os.path.join(self.data_path,"color","*.png"))]
        print("data path: ", self.data_path)
        print("data length: ", len(self.filenames))
        # self.rand_seed = 42
        # np.random.seed(self.rand_seed)
        # self.rand_seeds = np.random.randint(100, size=self.opt.max_epochs)
        # self.mean_matrix = np.zeros((self.opt.image_width, self.opt.image_height,3))
        # self.std_matrix = np.zeros((self.opt.image_width, self.opt.image_height,3))
        # for i in range(len(self.mean_matrix)):
        #     for j in range(len(self.mean_matrix[0])):
        #         self.mean_matrix[i,j,:] = [0.485, 0.456, 0.406]
        #         self.std_matrix[i,j,:] = [0.229, 0.224, 0.225]

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
        #depth = get_depth(self.data_path+'depth/'+self.filenames[idx]+".png", (self.opt.image_width, self.opt.image_height))
        #print(self.data_path+'color/'+self.filenames[idx]+'.png')
        color = cv2.resize(cv2.imread(os.path.join(self.data_path,'color',self.filenames[idx]+".png")), (self.opt.image_width, self.opt.image_height), interpolation = cv2.INTER_AREA)
        # color = ((color/255)-self.mean_matrix)/self.std_matrix
        label = cv2.resize(cv2.imread(os.path.join(self.data_path,'label',self.filenames[idx]+".png"), cv2.IMREAD_UNCHANGED), (self.opt.image_width, self.opt.image_height), interpolation = cv2.INTER_AREA)
        # img = np.concatenate((color, depth), axis=2)
        img = color
        img = rearrange(img, 'h w c -> c h w')
        label = self.encoding_label(label)
        label = rearrange(label, 'h w c -> c h w')

        return img.astype(np.float32), label.astype(np.float32)

    def __len__(self):
        return len(self.filenames)

    def rand_crop(self, img, size):
        np.random.seed(self.rand_seeds[0])
        buffer_h, buffer_w = h - size[0], w - size[1]
        rand_h = np.random.randint(buffer_h, size=1)[0]
        rand_w = np.random.randint(buffer_w, size=1)[0]
        c, h, w = img.shape
        img = img[rand_h:rand_h+size[0], rand_w:rand_w+size[1]]
        return img
