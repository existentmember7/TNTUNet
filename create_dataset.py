import numpy as np
import cv2
import json
import glob
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from einops import *
import csv
from matplotlib import pyplot as plt

## for preparing datasets
def get_segmantation(img_shape, label_path):
    spacing_label = get_spacing_label(label_path)
    img = get_masks(img_shape, spacing_label, 255)
    return img

def get_spacing_label(label_path):
    with open(label_path) as f:
        label = json.load(f)
        keyValList = ['spacing']
        spacing_labels = list(filter(lambda d: d['label'] in keyValList, label['shapes']))
        return spacing_labels

def get_masks(img_shape, labels, class_No):
    img = np.zeros((img_shape[0], img_shape[1], 1), dtype=np.uint8)
    for l in labels:
        points = np.array(l['points'], dtype=int)
        img = cv2.line(img, (points[0][0],points[0][1]), (points[1][0],points[1][1]), class_No, 10)
    return img

def generate_masks(path):
    img_path = path + 'color/'
    json_path = path + 'json/'
    mask_path = path + 'mask/'
    for file in glob.glob(img_path+'/*.png'):
        filename = file.split('/')[-1].split('.')[0]
        img_shape = cv2.imread(file).shape
        img = get_segmantation(img_shape, json_path+filename+'.json')
        cv2.imwrite(mask_path+filename+".png", img)

def mask2label(path, label_number):
    for img_path in glob.glob(path+'mask/*.png'):
        filename = img_path.split('/')[-1].split('.')[0]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img[img > 0] = label_number
        cv2.imwrite(path+"label/"+filename+".png", img)

def crop2split_data(path):
    folders = ['color', 'depth', 'label', 'mask']
    for folder in folders:
        for file in glob.glob(path+folder+'/*.png'):
            crop_and_split(file, (720, 720), path[:-1]+"_v2/"+folder+"/")

def crop_and_split(img_path, size, save_path):
    filename = img_path.split('/')[-1].split('.')[0]
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img1 = img[:size[0], :size[1]]
    img2 = img[:size[0], img.shape[1]-size[1]:]
    cv2.imwrite(save_path+filename+'_l.png', img1)
    cv2.imwrite(save_path+filename+'_r.png', img2)

def crop2split_data_3(path):
    folders = ['color', 'depth', 'label', 'mask']
    for folder in folders:
        for file in glob.glob(path+folder+'/*.png'):
            crop_and_split_3(file, (360, 360), path[:-4]+"_v3/"+folder+"/")

def crop_and_split_3(img_path, size, save_path):
    filename = img_path.split('/')[-1].split('.')[0]
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img1 = img[:size[0], :size[1]]
    img2 = img[:size[0], size[1]:]
    img3 = img[size[0]:, :size[1]] 
    img4 = img[size[0]:, size[1]:]
    cv2.imwrite(save_path+filename+'_1.png', img1)
    cv2.imwrite(save_path+filename+'_2.png', img2)
    cv2.imwrite(save_path+filename+'_3.png', img3)
    cv2.imwrite(save_path+filename+'_4.png', img4)



def get_depth(img_path, img_shape):
    img = cv2.resize(cv2.imread(img_path), img_shape, interpolation = cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,0]
    hsv = hsv.reshape(img_shape[0],img_shape[1], 1)
    return hsv

## end


# generate_masks('/media/han/D/aicenter_rebar_data/data/validation/')
# mask2label('/media/han/D/aicenter_rebar_data/data/validation/', 1)
# crop2split_data_3("/media/han/D/aicenter_rebar_data/data/validation_v2/")

# for ics competition proj1
base_dir = "/media/han/D/aicenter_rebar_data/ics/data_proj1_Tokaido_dataset/Tokaido_dataset/"

def get_full_file_path(filename):
    file = base_dir + filename.replace('\\','/')
    return file

def get_all_correct_file_path(row):
    file_list = []
    for i in range(4):
        file_list.append(get_full_file_path(row[i][1:]))
    for i in range(4,7):
        file_list.append(row[i])
    return file_list

def read_csv(filename):
    filenames = []
    with open(filename, "r") as file:
        for r in file.readlines():
            filenames.append(get_all_correct_file_path(r.split('\n')[0].split(',')))
    return filenames

def crop_and_split(img, filename, size, save_path):
    
    img1 = img[:size[0], :size[1]]
    img2 = img[:size[0], img.shape[1]-size[1]:]

    cv2.imwrite(save_path+filename+'_l.png', img1)
    cv2.imwrite(save_path+filename+'_r.png', img2)

def create_dataset():
    train_filenames_list = read_csv(base_dir + "files_train.csv")
    train_structural_component_path = "train_structural_component/"
    train_damage_path = "train_damage/"
    test_structural_component_path = "test_structural_component/"
    test_damage_path = "test_damage/"
    val_structural_component_path = "val_structural_component/"
    val_damage_path = "val_damage/"
    image_size = (640,360)
    count_structural_component = 0
    count_damage = 0
    for train_filenames in tqdm(train_filenames_list):
        # img
        filename = train_filenames[0].split('/')[-1].split('.')[0]
        img = cv2.imread(train_filenames[0], cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, image_size, interpolation = cv2.INTER_AREA)

        # label
        if train_filenames[5] == "True":
            depth = cv2.imread(train_filenames[3], cv2.IMREAD_UNCHANGED)
            label_bmp = cv2.imread(train_filenames[1], cv2.IMREAD_UNCHANGED)
            if count_structural_component % 10 == 0:
                crop_and_split(img, filename, (image_size[1],image_size[1]), base_dir + test_structural_component_path+"color/")
                crop_and_split(depth, filename, (image_size[1],image_size[1]), base_dir + test_structural_component_path+"depth/")
                crop_and_split(label_bmp, filename, (image_size[1],image_size[1]), base_dir + test_structural_component_path+"label/")
            elif count_structural_component % 10 == 1:
                crop_and_split(img, filename, (image_size[1],image_size[1]), base_dir + val_structural_component_path+"color/")
                crop_and_split(depth, filename, (image_size[1],image_size[1]), base_dir + val_structural_component_path+"depth/")
                crop_and_split(label_bmp, filename, (image_size[1],image_size[1]), base_dir + val_structural_component_path+"label/")
            else:
                crop_and_split(img, filename, (image_size[1],image_size[1]), base_dir + train_structural_component_path+"color/")
                crop_and_split(depth, filename, (image_size[1],image_size[1]), base_dir + train_structural_component_path+"depth/")
                crop_and_split(label_bmp, filename, (image_size[1],image_size[1]), base_dir + train_structural_component_path+"label/")
            count_structural_component += 1

        # label
        if train_filenames[6] == "True":
            label_bmp = cv2.imread(train_filenames[2], cv2.IMREAD_UNCHANGED)
            depth = cv2.imread(train_filenames[3], cv2.IMREAD_UNCHANGED)
            if count_damage % 10 == 0:
                crop_and_split(img, filename, (image_size[1],image_size[1]), base_dir + test_damage_path+"color/")
                crop_and_split(depth, filename, (image_size[1],image_size[1]), base_dir + test_damage_path+"depth/")
                crop_and_split(label_bmp, filename, (image_size[1],image_size[1]), base_dir + test_damage_path+"label/")
            elif count_damage % 10 == 1:
                crop_and_split(img, filename, (image_size[1],image_size[1]), base_dir + val_damage_path+"color/")
                crop_and_split(depth, filename, (image_size[1],image_size[1]), base_dir + val_damage_path+"depth/")
                crop_and_split(label_bmp, filename, (image_size[1],image_size[1]), base_dir + val_damage_path+"label/")
            else:
                crop_and_split(img, filename, (image_size[1],image_size[1]), base_dir + train_damage_path+"color/")
                crop_and_split(depth, filename, (image_size[1],image_size[1]), base_dir + train_damage_path+"depth/")
                crop_and_split(label_bmp, filename, (image_size[1],image_size[1]), base_dir + train_damage_path+"label/")
            count_damage += 1

# create_dataset()

# for ics competition proj2
base_dir = "/media/wisccitl/15afc964-cd4b-4320-bb71-364fba832bb1/han/data_proj2_QuakeCity/"

def get_full_file_path(filename, folder):
    file = base_dir + folder + filename
    return file

def get_all_correct_file_path(row):
    file_list = []
    # folder_list = ["image/","label/component/","label/crack/","label/spall/","label/rebar/","label/ds/","label/depth/"]
    # folder_list = ["image/","label/component/","label/depth/"]
    folder_list = ["image/","label/component/","label/crack/","label/spall/","label/rebar/","label/depth/"]
    for folder in folder_list:
        file_list.append(get_full_file_path(row[0], folder))
    return file_list

def create_dataset_all():
    # ignore, wall, beam, column, window frame, window pane, balcony, slab
    class_list = [[[70,70,70],[150,150,202],[100,186,198],[186,183,167],[133,255,255],[206,192,192],[160,80,32],[1,134,193]],[[0,0,255]],[[203,192,255]],[[50,225,255]]]
    # class_list = [[70,70,70],[150,150,202],[100,186,198],[183,186,167],[133,255,255],[206,192,192],[160,80,32],[1,134,193]]
    with open('/home/user/Documents/han/data/train.csv', newline='\n') as csvfile:
        spamreader = csv.reader(csvfile)
        count = 0# for ics competition proj1

def get_full_file_path(filename):
    file = base_dir + filename.replace('\\','/')
    return file

def get_all_correct_file_path(row):
    file_list = []
    for i in range(4):
        file_list.append(get_full_file_path(row[i][1:]))
    for i in range(4,7):
        file_list.append(row[i])
    return file_list

def read_csv(filename):
    filenames = []
    with open(filename, "r") as file:
        for r in file.readlines():
            filenames.append(get_all_correct_file_path(r.split('\n')[0].split(',')))
    return filenames

def crop_and_split(img, filename, size, save_path):
    
    img1 = img[:size[0], :size[1]]
    img2 = img[:size[0], img.shape[1]-size[1]:]

    cv2.imwrite(save_path+filename+'_l.png', img1)
    cv2.imwrite(save_path+filename+'_r.png', img2)

def create_dataset():
    train_filenames_list = read_csv(base_dir + "files_train.csv")
    train_structural_component_path = "train_structural_component/"
    train_damage_path = "train_damage/"
    test_structural_component_path = "test_structural_component/"
    test_damage_path = "test_damage/"
    val_structural_component_path = "val_structural_component/"
    val_damage_path = "val_damage/"
    image_size = (640,360)
    count_structural_component = 0
    count_damage = 0
    for train_filenames in tqdm(train_filenames_list):
        # img
        filename = train_filenames[0].split('/')[-1].split('.')[0]
        img = cv2.imread(train_filenames[0], cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, image_size, interpolation = cv2.INTER_AREA)

        # label
        if train_filenames[5] == "True":
            depth = cv2.imread(train_filenames[3], cv2.IMREAD_UNCHANGED)
            label_bmp = cv2.imread(train_filenames[1], cv2.IMREAD_UNCHANGED)
            if count_structural_component % 10 == 0:
                crop_and_split(img, filename, (image_size[1],image_size[1]), base_dir + test_structural_component_path+"color/")
                crop_and_split(depth, filename, (image_size[1],image_size[1]), base_dir + test_structural_component_path+"depth/")
                crop_and_split(label_bmp, filename, (image_size[1],image_size[1]), base_dir + test_structural_component_path+"label/")
            elif count_structural_component % 10 == 1:
                crop_and_split(img, filename, (image_size[1],image_size[1]), base_dir + val_structural_component_path+"color/")
                crop_and_split(depth, filename, (image_size[1],image_size[1]), base_dir + val_structural_component_path+"depth/")
                crop_and_split(label_bmp, filename, (image_size[1],image_size[1]), base_dir + val_structural_component_path+"label/")
            else:
                crop_and_split(img, filename, (image_size[1],image_size[1]), base_dir + train_structural_component_path+"color/")
                crop_and_split(depth, filename, (image_size[1],image_size[1]), base_dir + train_structural_component_path+"depth/")
                crop_and_split(label_bmp, filename, (image_size[1],image_size[1]), base_dir + train_structural_component_path+"label/")
            count_structural_component += 1

        # label
        if train_filenames[6] == "True":
            label_bmp = cv2.imread(train_filenames[2], cv2.IMREAD_UNCHANGED)
            depth = cv2.imread(train_filenames[3], cv2.IMREAD_UNCHANGED)
            if count_damage % 10 == 0:
                crop_and_split(img, filename, (image_size[1],image_size[1]), base_dir + test_damage_path+"color/")
                crop_and_split(depth, filename, (image_size[1],image_size[1]), base_dir + test_damage_path+"depth/")
                crop_and_split(label_bmp, filename, (image_size[1],image_size[1]), base_dir + test_damage_path+"label/")
            elif count_damage % 10 == 1:
                crop_and_split(img, filename, (image_size[1],image_size[1]), base_dir + val_damage_path+"color/")
                crop_and_split(depth, filename, (image_size[1],image_size[1]), base_dir + val_damage_path+"depth/")
                crop_and_split(label_bmp, filename, (image_size[1],image_size[1]), base_dir + val_damage_path+"label/")
            else:
                crop_and_split(img, filename, (image_size[1],image_size[1]), base_dir + train_damage_path+"color/")
                crop_and_split(depth, filename, (image_size[1],image_size[1]), base_dir + train_damage_path+"depth/")
                crop_and_split(label_bmp, filename, (image_size[1],image_size[1]), base_dir + train_damage_path+"label/")
            count_damage += 1

# create_dataset()

# for ics competition proj
        for row in tqdm(spamreader):
            file_list = get_all_correct_file_path(row)
            filename = file_list[0].split('/')[-1].split('.')[0]
            img = cv2.imread(file_list[0], cv2.IMREAD_UNCHANGED)
            depth = cv2.imread(file_list[5], cv2.IMREAD_UNCHANGED)
            label = np.zeros((img.shape[0],img.shape[1]))
            count_label = 0
            for j in range(1,len(file_list)-1):
                annotation = cv2.imread(file_list[j], cv2.IMREAD_UNCHANGED)
                for i in range(len(class_list[j-1])):
                    label[(annotation==class_list[j-1][i]).all(2)] = count_label
                    count_label += 1
        
                # img[(annotation==class_list[0]).all(2)] = [0,0,0] # Make the background bllack

            if count%10 == 0 or count%10 == 1:
                crop_and_split(img,filename,(1080,1080),base_dir+"test/color/")
                #crop_and_split(depth,filename,(1080,1080),base_dir+"test/depth/")
                crop_and_split(label,filename,(1080,1080),base_dir+"test/label/")
                #img[(annotation==class_list[0][0]).all(2)] == [0,0,0]
                #crop_and_split(img,filename,(1080,1080),base_dir+"val/color/")
                #crop_and_split(label,filename,(1080,1080),base_dir+"val/label/")
            #elif count%10 == 1:
            #    crop_and_split(img,filename,(1080,1080),base_dir+"val/color/")
            #    crop_and_split(depth,filename,(1080,1080),base_dir+"val/depth/")
            #    crop_and_split(label,filename,(1080,1080),base_dir+"val/label/")
            else:
                crop_and_split(img,filename,(1080,1080),base_dir+"train/color/")
                crop_and_split(depth,filename,(1080,1080),base_dir+"train/depth/")
                crop_and_split(label,filename,(1080,1080),base_dir+"train/label/")
            count += 1

# create_dataset_all()

def get_full_file_path(filename, folder):
    file = base_dir + folder + filename
    return file

def get_all_correct_file_path(row):
    file_list = []
    folder_list = ["image/","label/component/","label/depth/"]
    for folder in folder_list:
        file_list.append(get_full_file_path(row[0], folder))
    return file_list

def create_dataset_component():
    # ignore, wall, beam, column, window frame, window pane, balcony, slab
    class_list = [[70,70,70],[150,150,202],[100,186,198],[186,183,167],[133,255,255],[206,192,192],[160,80,32],[1,134,193]]
    with open('/media/wisccitl/15afc964-cd4b-4320-bb71-364fba832bb1/han/data_proj2_QuakeCity/train.csv', newline='\n') as csvfile:
        spamreader = csv.reader(csvfile)
        count = 0
        for row in tqdm(spamreader):
            file_list = get_all_correct_file_path(row)
            filename = file_list[0].split('/')[-1].split('.')[0]
            img = cv2.imread(file_list[0], cv2.IMREAD_UNCHANGED)

            #depth = cv2.imread(file_list[5], cv2.IMREAD_UNCHANGED)
            label = np.zeros((img.shape[0],img.shape[1]))
            annotation = cv2.imread(file_list[1], cv2.IMREAD_UNCHANGED)
            for i in range(len(class_list)):
                label[(annotation==class_list[i]).all(2)] = i
        
                # img[(annotation==class_list[0]).all(2)] = [0,0,0] # Make the background bllack

            if count%10 == 0 or count%10 == 1:
                crop_and_split(img,filename,(1080,1080),base_dir+"test/color/")
                #crop_and_split(depth,filename,(1080,1080),base_dir+"test/depth/")
                crop_and_split(label,filename,(1080,1080),base_dir+"test/label/")
                img[(annotation==class_list[0]).all(2)] = [0,0,0]
                crop_and_split(img,filename,(1080,1080),base_dir+"val/color/")
                crop_and_split(label,filename,(1080,1080),base_dir+"val/label/")
            # elif count%10 == 1:
            #    crop_and_split(img,filename,(1080,1080),base_dir+"val/color/")
            #    crop_and_split(depth,filename,(1080,1080),base_dir+"val/depth/")
            #    crop_and_split(label,filename,(1080,1080),base_dir+"val/label/")
            else:
                img[(annotation==class_list[0]).all(2)] = [0,0,0]
                crop_and_split(img,filename,(1080,1080),base_dir+"train/color/")
                #crop_and_split(depth,filename,(1080,1080),base_dir+"train/depth/")
                crop_and_split(label,filename,(1080,1080),base_dir+"train/label/")
            count += 1

create_dataset_component()

# for creating destroy dataset
def get_all_correct_file_path_destroy(row):
    file_list = []
    # folder_list = ["image/","label/component/","label/crack/","label/spall/","label/rebar/","label/ds/","label/depth/"]
    # folder_list = ["image/","label/component/","label/depth/"]
    folder_list = ["image/", "label/crack/","label/spall/","label/rebar/","label/depth/"]
    for folder in folder_list:
        file_list.append(get_full_file_path(row[0], folder))
    return file_list

def create_destroy_dataset():
    # ignore, crack, spall, balcony, slab
    class_list = [[0,0,255],[203,192,255],[50,225,255]]
    with open('/home/user/Documents/han/data/train.csv', newline='\n') as csvfile:
        spamreader = csv.reader(csvfile)
        count = 0
        for row in tqdm(spamreader):
            file_list = get_all_correct_file_path_destroy(row)
            filename = file_list[0].split('/')[-1].split('.')[0]
            img = cv2.imread(file_list[0], cv2.IMREAD_UNCHANGED)
            depth = cv2.imread(file_list[4], cv2.IMREAD_UNCHANGED)
            label = np.zeros((img.shape[0], img.shape[1]))

            for i in range(1,len(file_list)-1):
                annotation = cv2.imread(file_list[i], cv2.IMREAD_UNCHANGED)
                label[(annotation == class_list[i-1]).all(2)] = i
            
            if count%10 == 0 or count%10 == 1:
                crop_and_split(img, filename, (1080,1080), base_dir+"test/color/")
                crop_and_split(depth, filename, (1080,1080), base_dir+"test/depth/")
                crop_and_split(label, filename, (1080,1080), base_dir+"test/label/")
            else:
                crop_and_split(img, filename, (1080,1080), base_dir+"train/color/")
                crop_and_split(depth, filename, (1080,1080), base_dir+"train/depth/")
                crop_and_split(label, filename, (1080,1080), base_dir+"train/label/")
            count += 1
# create_destroy_dataset()
# for creating destroy dataset
def get_all_correct_file_path_slab(row):
    file_list = []
    folder_list = ["image/","label/component/","label/depth/"]
    for folder in folder_list:
        file_list.append(get_full_file_path(row[0], folder))
    return file_list

def create_slab_dataset():
    # ignore, slab
    class_list = [[1,134,193]]
    with open('/home/user/Documents/han/data/train.csv', newline='\n') as csvfile:
        spamreader = csv.reader(csvfile)
        count = 0
        for row in tqdm(spamreader):
            file_list = get_all_correct_file_path_slab(row)
            filename = file_list[0].split('/')[-1].split('.')[0]
            img = cv2.imread(file_list[0], cv2.IMREAD_UNCHANGED)
            depth = cv2.imread(file_list[2], cv2.IMREAD_UNCHANGED)
            label = np.zeros((img.shape[0], img.shape[1]))

            for i in range(1,len(file_list)-1):
                annotation = cv2.imread(file_list[i], cv2.IMREAD_UNCHANGED)
                label[(annotation == class_list[i-1]).all(2)] = i
            
            if count%10 == 0 or count%10 == 1:
                crop_and_split(img, filename, (1080,1080), base_dir+"test/color/")
                crop_and_split(depth, filename, (1080,1080), base_dir+"test/depth/")
                crop_and_split(label, filename, (1080,1080), base_dir+"test/label/")
            else:
                crop_and_split(img, filename, (1080,1080), base_dir+"train/color/")
                crop_and_split(depth, filename, (1080,1080), base_dir+"train/depth/")
                crop_and_split(label, filename, (1080,1080), base_dir+"train/label/")
            count += 1
#create_slab_dataset()

def get_full_file_path(filename, folder):
    file = base_dir + folder + filename
    return file

def get_all_correct_file_path_for_test(row):
    file_list = []
    folder_list = ["image/"]
    for folder in folder_list:
        file_list.append(get_full_file_path(row[0], folder))
    return file_list

def create_dataset_component_for_testing():
    # ignore, wall, beam, column, window frame, window pane, balcony, slab
    class_list = [[70,70,70],[150,150,202],[100,186,198],[183,186,167],[133,255,255],[206,192,192],[160,80,32],[1,134,193]]
    with open('/media/wisccitl/15afc964-cd4b-4320-bb71-364fba832bb1/han/data_proj2_QuakeCity/test.csv', newline='\n') as csvfile:
        spamreader = csv.reader(csvfile)
        count = 0
        for row in tqdm(spamreader):
            file_list = get_all_correct_file_path(row)
            filename = file_list[0].split('/')[-1].split('.')[0]
            print(file_list[0])
            img = cv2.imread(file_list[0], cv2.IMREAD_UNCHANGED)

            crop_and_split(img,filename,(1080,1080),base_dir+"real_test/color/")
            img = img[:,:,0]
            crop_and_split(img,filename,(1080,1080),base_dir+"real_test/label/")

            count += 1

# create_dataset_component_for_testing()