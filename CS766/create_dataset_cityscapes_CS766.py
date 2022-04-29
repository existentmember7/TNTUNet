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

# labels = [
#     #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
#     Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
#     Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
#     Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
#     Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
#     Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
#     Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
#     Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
#     Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
#     Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
#     Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
#     Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
#     Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
#     Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
#     Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
#     Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
#     Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
#     Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
#     Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
#     Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
#     Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
#     Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
#     Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
#     Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
#     Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
#     Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
#     Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
#     Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
#     Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
#     Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
#     Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
# ]


# _cmap = {
#     0: (128, 128, 128),    # Sky
#     1: (128, 0, 0),        # Building
#     2: (128, 64, 128),     # Road
#     3: (0, 0, 192),        # Sidewalk
#     4: (64, 64, 128),      # Fence
#     5: (128, 128, 0),      # Vegetation
#     6: (192, 192, 128),    # Pole
#     7: (64, 0, 128),       # Car
#     8: (192, 128, 128),    # Sign
#     9: (64, 64, 0),        # Pedestrian
#     10: (0, 128, 192),     # Cyclist
#     11: (0, 0, 0)          # Void
#     }

# _mask_labels = {0: 'Sky', 1: 'Building', 2: 'Road', 3: 'Sidewalk',
#                 4: 'Fence', 5: 'Vegetation', 6: 'Pole', 7: 'Car',
#                 8: 'Sign', 9: 'Pedestrian', 10: 'Cyclist', 11: 'void'}

# base_dir = "D:\\han\\KITTI_new\\origin"
# destination_dir = "D:\\han\\KITTI_new\\dataset"
# folder = "train"

# if os.path.exists(destination_dir) != True:
#     os.mkdir(destination_dir)
#     os.mkdir(osp.join(destination_dir,"train"))
#     os.mkdir(osp.join(destination_dir,"val"))
#     os.mkdir(osp.join(osp.join(destination_dir,"train"),"color"))
#     os.mkdir(osp.join(osp.join(destination_dir,"train"),"label"))
#     os.mkdir(osp.join(osp.join(destination_dir,"train"),"rgb"))
#     os.mkdir(osp.join(osp.join(destination_dir,"val"),"color"))
#     os.mkdir(osp.join(osp.join(destination_dir,"val"),"label"))
#     os.mkdir(osp.join(osp.join(destination_dir,"val"),"rgb"))

# count = 0
# for filepath in tqdm.tqdm(glob(osp.join(osp.join(osp.join(base_dir,folder),"RGB"),"*.png"))):
#     filename = get_filename(filepath)
#     # print(filename)
#     # if count % 5 != 0:
#     shutil.copyfile(filepath, osp.join(osp.join(osp.join(destination_dir,folder),"color"),filename))
#     gt = cv2.imread(osp.join(osp.join(osp.join(base_dir,folder),"GT"),filename))
#     gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
#     label = np.zeros(gt.shape)
#     for key in _cmap:
#         color = np.array([ _cmap[key][0],  _cmap[key][1],  _cmap[key][2]])
#         for h in range(gt.shape[0]):
#             for w in range(gt.shape[1]):
#                 if (gt[h,w] == color).all():
#                     label[h,w] = key
#     # print(np.unique(label))
#     label = (np.sum(label, axis=2)/3).astype(int)
#     cv2.imwrite(osp.join(osp.join(osp.join(destination_dir,folder),"label"),filename),label)
#     color = np.zeros(gt.shape)
#     for key in _cmap:
#         color[label == key, 0] = _cmap[key][0]
#         color[label == key, 1] = _cmap[key][1]
#         color[label == key, 2] = _cmap[key][2]
    
#     color_temp = color.copy()
#     color[:,:,0] = color_temp[:,:,2]
#     color[:,:,2] = color_temp[:,:,0]
#     cv2.imwrite(osp.join(osp.join(osp.join(destination_dir,folder),"rgb"),filename),color)
#     # exit(-1)

#     # shutil.copyfile(osp.join(osp.join(base_dir,"semantic"), filename), osp.join(osp.join(osp.join(destination_dir,"train"),"label"),filename))
#     # else:
#     #     shutil.copyfile(filepath, osp.join(osp.join(osp.join(destination_dir,"val"),"color"),filename))
#     #     shutil.copyfile(osp.join(osp.join(base_dir,"semantic"), filename), osp.join(osp.join(osp.join(destination_dir,"val"),"label"),filename))
    
#     count += 1

base_dir = "D:\\han\\gtFine_trainvaltest\\gtFine"
destination_dir = "D:\\han\\gtFine_trainvaltest\\dataset"
folder = "train\\*\\*_instanceIds.png"

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

for filepath in glob(osp.join(base_dir, folder)):
    filename = get_filename(filepath)
    shutil.copyfile(filepath, osp.join(osp.join(osp.join(destination_dir,folder),"color"),filename))
    shutil.copyfile(filepath, osp.join(osp.join(osp.join(destination_dir,folder),"label"),filename))


