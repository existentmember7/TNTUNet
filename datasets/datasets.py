from numpy.lib import utils
from torch.utils.data import Dataset
import cv2
import glob
import os
from utils import *
import numpy as np
from einops import *

class CustomDataset(Dataset):

    def __init__(self, _opt):
        self.opt = _opt
        self.filenames = [os.path.basename(f).split('.')[0] for f in glob.glob(self.opt.training_data_path+"color/*.png")]

    def encoding_label(self, labels):
        temp_labels = np.zeros((labels.shape[0], labels.shape[1], self.opt.num_classes))
        temp_labels[labels != 0] = [0,1]
        temp_labels[labels == 0] = [1,0]
        return temp_labels


    def __getitem__(self, idx):
        depth = get_depth(self.opt.training_data_path+'depth/'+self.filenames[idx]+".png", (self.opt.image_width, self.opt.image_height))
        color = cv2.resize(cv2.imread(self.opt.training_data_path+'color/'+self.filenames[idx]+".png"), (self.opt.image_width, self.opt.image_height), interpolation = cv2.INTER_AREA)
        label = cv2.resize(cv2.imread(self.opt.training_data_path+'label/'+self.filenames[idx]+".png", cv2.IMREAD_UNCHANGED), (self.opt.image_width, self.opt.image_height), interpolation = cv2.INTER_AREA)
        img = np.concatenate((color, depth), axis=2)
        img = rearrange(img, 'h w c -> c h w')
        label = self.encoding_label(label)
        label = rearrange(label, 'h w c -> c h w')

        return img.astype(np.float32), label.astype(np.float32)

    def __len__(self):
        return len(self.filenames)
