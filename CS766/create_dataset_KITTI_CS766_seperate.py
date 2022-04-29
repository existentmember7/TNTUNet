from glob import glob
import os
import os.path as osp
import shutil
import tqdm
import numpy as np
import cv2

# function
def get_filename(path):
    filename = path.split('\\')[-1]
    return filename

_cmap = {
        0: (128, 128, 128),    # Sky
        1: (128, 0, 0),        # Building
        2: (128, 64, 128),     # Road
        3: (0, 0, 192),        # Sidewalk
        4: (64, 64, 128),      # Fence
        5: (128, 128, 0),      # Vegetation
        6: (192, 192, 128),    # Pole
        7: (64, 0, 128),       # Car
        8: (192, 128, 128),    # Sign
        9: (64, 64, 0),        # Pedestrian
        10: (0, 128, 192),     # Cyclist
        11: (0, 0, 0)          # Void
    }
_mask_labels = {0: 'Sky', 1: 'Building', 2: 'Road', 3: 'Sidewalk',
                4: 'Fence', 5: 'Vegetation', 6: 'Pole', 7: 'Car',
                8: 'Sign', 9: 'Pedestrian', 10: 'Cyclist', 11: 'void'}

base_dir = "D:\\han\\KITTI_new\\origin"
destination_dir = "D:\\han\\KITTI_new\\big_dataset"
folder = "val"

class_list = [0,1,2,3,5,7]
# class_list = [4,6,8,9,10,11]

if os.path.exists(destination_dir) != True:
    os.mkdir(destination_dir)
    os.mkdir(osp.join(destination_dir,"train"))
    os.mkdir(osp.join(destination_dir,"val"))
    os.mkdir(osp.join(osp.join(destination_dir,"train"),"color"))
    os.mkdir(osp.join(osp.join(destination_dir,"train"),"label"))
    os.mkdir(osp.join(osp.join(destination_dir,"train"),"rgb"))
    os.mkdir(osp.join(osp.join(destination_dir,"val"),"color"))
    os.mkdir(osp.join(osp.join(destination_dir,"val"),"label"))
    os.mkdir(osp.join(osp.join(destination_dir,"val"),"rgb"))

count = 0
for filepath in tqdm.tqdm(glob(osp.join(osp.join(osp.join(base_dir,folder),"RGB"),"*.png"))):
    filename = get_filename(filepath)
    # print(filename)
    # if count % 5 != 0:
    shutil.copyfile(filepath, osp.join(osp.join(osp.join(destination_dir,folder),"color"),filename))
    gt = cv2.imread(osp.join(osp.join(osp.join(base_dir,folder),"GT"),filename))
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
    label = np.zeros(gt.shape)
    for key in _cmap:
        color = np.array([ _cmap[key][0],  _cmap[key][1],  _cmap[key][2]])
        for h in range(gt.shape[0]):
            for w in range(gt.shape[1]):
                if (gt[h,w] == color).all():
                    if key in class_list:
                        label[h,w] = class_list.index(key)
                    else:
                        label[h,w] = 5
    # print(np.unique(label))
    label = (np.sum(label, axis=2)/3).astype(int)
    cv2.imwrite(osp.join(osp.join(osp.join(destination_dir,folder),"label"),filename), label)
    color = np.zeros(gt.shape)
    for key in _cmap:
        if key in class_list:
            color[label == class_list.index(key), 0] = _cmap[key][0]
            color[label == class_list.index(key), 1] = _cmap[key][1]
            color[label == class_list.index(key), 2] = _cmap[key][2]
        else:
            color[label == 5, 0] = 0
            color[label == 5, 1] = 0
            color[label == 5, 2] = 0
    
    color_temp = color.copy()
    color[:,:,0] = color_temp[:,:,2]
    color[:,:,2] = color_temp[:,:,0]
    cv2.imwrite(osp.join(osp.join(osp.join(destination_dir,folder),"rgb"),filename), color)
    # exit(-1)

    # shutil.copyfile(osp.join(osp.join(base_dir,"semantic"), filename), osp.join(osp.join(osp.join(destination_dir,"train"),"label"),filename))
    # else:
    #     shutil.copyfile(filepath, osp.join(osp.join(osp.join(destination_dir,"val"),"color"),filename))
    #     shutil.copyfile(osp.join(osp.join(base_dir,"semantic"), filename), osp.join(osp.join(osp.join(destination_dir,"val"),"label"),filename))
    
    count += 1


