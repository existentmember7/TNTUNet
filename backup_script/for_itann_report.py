import cv2
import numpy as np
import os
import glob

prediction_path = '/home/wisccitl/Documents/han/TNTUNet/test_result_img/'
data_path = '/media/wisccitl/15afc964-cd4b-4320-bb71-364fba832bb1/han/itann_final_project/dataset/val/'

def result(filename):
    background = cv2.imread(os.path.join(data_path, 'color',filename))
    overlay_label = cv2.imread(os.path.join(data_path, 'label',filename))
    overlay_prediction = cv2.imread(os.path.join(prediction_path,filename))
    overlay_label = cv2.resize(overlay_label,(256,256))
    background = cv2.resize(background,(256,256))
    overlay_label[:,:,2][overlay_label[:,:,0] > 0] = 255
    overlay_label[:,:,0][overlay_label[:,:,0] > 0] = 0
    overlay_prediction[:,:,2][overlay_prediction[:,:,0] > 0] = 255
    overlay_prediction[:,:,0][overlay_prediction[:,:,0] > 0] = 0

    added_image_prediction = cv2.addWeighted(background,1.0,overlay_prediction,1.0,0)
    added_image_label = cv2.addWeighted(background,1.0,overlay_label,1.0,0)

    cv2.imwrite('/home/wisccitl/Documents/han/TNTUNet/itann_result/prediction/combined_pred_'+filename, added_image_prediction)
    cv2.imwrite('/home/wisccitl/Documents/han/TNTUNet/itann_result/label/combined_label_'+filename, added_image_label)

count = 0
for file in glob.glob(prediction_path+'*png'):
    print(count)
    count += 1
    filename = file.split('/')[-1]
    result(filename)
