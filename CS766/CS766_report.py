import cv2
import glob
import os.path as osp
import numpy as np

colors_path = "D:\\han\\KITTI_new\\dataset\\val\\color"
colors_paths = glob.glob(osp.join(colors_path, "*.png"))
pred_path = "C:\\Users\\whan59\\Desktop\\TNTUNet\\results\\23-04-2022_11-13-19_bce+dice"
pred_paths = glob.glob(osp.join(pred_path, "*.png"))

img_array = []
for i in range(len(colors_paths)):
    color = cv2.imread(colors_paths[i])
    pred = cv2.imread(pred_paths[i])
    pred = cv2.resize(pred, (color.shape[1],color.shape[0]), interpolation = cv2.INTER_AREA)
    final = np.append(color, pred, axis=0)
    size = (final.shape[1],final.shape[0])
    img_array.append(final)

out = cv2.VideoWriter('CS766_project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 7, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()