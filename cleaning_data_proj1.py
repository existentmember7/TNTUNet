import numpy as np
import csv
import cv2
import os.path
from tqdm import tqdm 

base_dir = '/media/han/D/download/data_proj1_Tokaido_dataset/Tokaido_dataset'

def check_all_file():

    with open('/media/han/D/download/data_proj1_Tokaido_dataset/Tokaido_dataset/files_train.csv', newline='\n') as csvfile:
        spamreader = csv.reader(csvfile)
        count = 0
        count_err = 0
        for row in tqdm(spamreader):
            err = False
            # print(count)
            count+=1
            file_list = get_all_correct_file_path(row)
            if row[5] == 'True':
                for file in [file_list[0],file_list[1], file_list[3]]:
                    if not check_file_exist(file):
                        # print(file)
                        err = True
            if row[6] == 'True':
                for file in [file_list[0],file_list[2], file_list[3]]:
                    if not check_file_exist(file):
                        # print(file)
                        err = True
            if err:
                count_err += 1
        print("error file: " + str(count_err))

def check_class_distributiuon():

    F_distribition = np.zeros((9))
    G_distribition = np.zeros((4))

    count_F = 0
    count_G = 0

    with open('/media/han/D/download/data_proj1_Tokaido_dataset/Tokaido_dataset/files_train.csv', newline='\n') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in tqdm(spamreader):
            file_list = get_all_correct_file_path(row)
            if row[5] == 'True':
                count_F += 1
                if check_file_exist(file_list[1]):
                    image = cv2.imread(file_list[1], 0)
                    for i in range(F_distribition.shape[0]):
                        if i in image:
                            F_distribition[i] += 1
            if row[6] == 'True':
                count_G += 1
                if check_file_exist(file_list[2]):
                    for i in range(G_distribition.shape[0]):
                        if i in image:
                            G_distribition[i] += 1
    
    print("Distribution: ")
    print("F distribition = ", F_distribition, ", Total = ", count_F)
    print("G distribition = ", G_distribition, ", Total = ", count_G)

def check_class_distributiuon_puretex():

    G_distribition = np.zeros((4))
    
    count_G = 0

    with open('/media/han/D/download/data_proj1_Tokaido_dataset/Tokaido_dataset/files_puretex_train.csv', newline='\n') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in tqdm(spamreader):
            count_G += 1
            file_1 = get_full_file_path(row[0][1:])
            file_2 = get_full_file_path(row[1][1:])
            if check_file_exist(file_1) and check_file_exist(file_2):
                image = cv2.imread(file_2, 0)
                for i in range(G_distribition.shape[0]):
                    if i in image:
                        G_distribition[i] += 1
    
    print("Distribution: ")
    print("G distribition = ", G_distribition, ", Total = ", count_G)

def check_file_exist(file):
    if os.path.isfile(file) == True:
        return True
    return False

def get_full_file_path(filename):
    file = base_dir + filename.replace('\\','/')
    return file

def get_all_correct_file_path(row):
    file_list = []
    for i in range(4):
        file_list.append(get_full_file_path(row[i][1:]))
    return file_list

def read_bmp(file):
    image= cv2.imread(file,0)
    return image

# check_all_file()
check_class_distributiuon()
check_class_distributiuon_puretex()
