from numpy.lib import utils
from torch.utils.data import Dataset
import cv2
import glob
import os
from utils import *

class CustomDataset(Dataset):

    def __init__(self, _opt):
        self.opt = _opt
        self.filenames = [os.path.basename(f).split('.')[0] for f in glob.glob(self.opt.training_data_path+"color/*.png")]


    def __getitem__(self, idx):
        depth = get_depth(self.opt.training_data_path+'depth/'+self.filenames[idx]+".png", (self.opt.image_width, self.opt.image_height))
        color = cv2.resize(cv2.imread(self.opt.training_data_path+'color/'+self.filenames[idx]+".png"), (self.opt.image_width, self.opt.image_height), interpolation = cv2.INTER_AREA)
        label = cv2.resize(cv2.imread(self.opt.training_data_path+'label/'+self.filenames[idx]+".png", cv2.IMREAD_UNCHANGED), (self.opt.image_width, self.opt.image_height), interpolation = cv2.INTER_AREA)
        img = np.concatenate((color, depth), axis=2)

        return img, label

    def __len__(self):
        return len(self.filenames)
