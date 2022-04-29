import numpy as np
import csv
import cv2
import os.path
from tqdm import tqdm 

base_dir = '/media/han/D/download/data_proj2_QuakeCity/'

def check_all_file():

    with open('/media/han/D/download/data_proj2_QuakeCity/train.csv', newline='\n') as csvfile:
        spamreader = csv.reader(csvfile)
        count = 0
        count_err = 0
        for row in tqdm(spamreader):
            err = False
            # print(count)
            count+=1
            file_list = get_all_correct_file_path(row)
            for file in file_list:
                if not check_file_exist(file):
                    # print(file)
                    err = True
            if err:
                count_err += 1
        print("error file: " + str(count_err))

def check_class_distributiuon():

    distribution_list = []

    C_distribition = np.zeros((8))
    Dc_distribition = np.zeros((2))
    Ds_distribition = np.zeros((2))
    Dr_distribition = np.zeros((2))
    DS_distribition = np.zeros((5))

    distribution_list.append(C_distribition)
    distribution_list.append(Dc_distribition)
    distribution_list.append(Ds_distribition)
    distribution_list.append(Dr_distribition)
    distribution_list.append(DS_distribition)

    count_list = [0,0,0,0,0]


    # classes_list = [
    #     [[202,150,150],[198,186,100],[167,183,186],[255,255,133],[192,192,206],[32,80,160],[193,134,1],[70,70,70]],
    #     [[255,0,0],[0,0,0]],
    #     [[255, 192, 203],[0,0,0]],
    #     [[255, 255, 50],[0,0,0]],
    #     [[0,255,0],[150,250,0],[255,225,50],[255,0,0],[128,128,128]]
    #     ]
    classes_list = [
        [[150,150,202],[100,186,198],[186,183,167],[133,255,255],[206,192,192],[160,80,32],[1,134,193],[70,70,70]],
        [[0,0,255],[0,0,0]],
        [[203, 192, 255],[0,0,0]],
        [[50, 255, 255],[0,0,0]],
        [[0,255,0],[0,250,150],[50,225,255],[0,0,255],[128,128,128]]
        ]

    with open('/media/han/D/download/data_proj2_QuakeCity/train.csv', newline='\n') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in tqdm(spamreader):
            file_list = get_all_correct_file_path(row)
            count  = 3
            for file in file_list[4:5]:
                img = cv2.imread(file)
                count_list[count]+=1
                for i in range(distribution_list[count].shape[0]):
                    if (classes_list[count][i] == img).all(2).any():
                        # print(classes_list[count][i])
                        distribution_list[count][i] += 1
                count += 1


            
    
    print("Distribution: ")
    print("C distribition = ", distribution_list[0], ", Total = ", count_list[0])
    print("Dc distribition = ", distribution_list[1], ", Total = ", count_list[1])
    print("Ds distribition = ", distribution_list[2], ", Total = ", count_list[2])
    print("Dr distribition = ", distribution_list[3], ", Total = ", count_list[3])
    print("DS distribition = ", distribution_list[4], ", Total = ", count_list[4])


def check_file_exist(file):
    if os.path.isfile(file) == True:
        return True
    return False

def get_full_file_path(filename, folder):
    file = base_dir + folder + filename
    return file

def get_all_correct_file_path(row):
    file_list = []
    folder_list = ["image/","label/component/","label/crack/","label/spall/","label/rebar/","label/ds/","label/depth/"]
    for folder in folder_list:
        file_list.append(get_full_file_path(row[0], folder))
    return file_list

def read_bmp(file):
    image = cv2.imread(file,0)
    return image

def vis():
    with open('/media/han/D/download/data_proj2_QuakeCity/train.csv', newline='\n') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in tqdm(spamreader):
            file_list = get_all_correct_file_path(row)
            img = cv2.imread(file_list[0])
            crack = cv2.imread(file_list[2])
            image = combine_images(img, crack)
            cv2.imshow("image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            exit(-1)

def combine_images(img, label):
    l = (label!=[0,0,0])
    img[l] = label[l]
    return img

vis()

# check_all_file()
# check_class_distributiuon()


