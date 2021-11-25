import numpy as np
import json
import cv2
from os.path import exists
from tqdm import tqdm
import csv
import glob
import os
from numpy import genfromtxt

dataset_type = "val"
base_dir = os.path.join('D:\han\itann',dataset_type)
new_dir = os.path.join('D:\han\itann\dataset',dataset_type)

def read_csv():
    path ='categories.csv'

    new_cate_names = None
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            new_cate_names = row[3:-1]
            break
    # print(new_cate_names)
    categories = genfromtxt(path, delimiter=',')
    categories = categories[categories[:,-1] > 0][1:,:-1]
    # print(categories.shape)
    categories_dict = {}

    for i in range(categories.shape[0]):
        for j in range(3,categories.shape[1]):
            if categories[i,j] == 1:
                categories_dict[int(categories[i,1])] = {}
                categories_dict[int(categories[i,1])]['cate_id'] = j-3
                categories_dict[int(categories[i,1])]['name'] = new_cate_names[j-3]
                break
    # print(categories_dict[1566])
    # exit(-1)
    return categories_dict

    # for i in range(3,categories.shape[1]):
    #     categories_dict[i-3] = {}
    #     categories_dict[i-3]['cate_ids'] = []
    #     categories_dict[i-3]['name'] = new_cate_names[i-3]
    #     for j in range(categories.shape[0]):
    #         if categories[j,i] == 1:
    #             categories_dict[i-3]['cate_ids'].append(int(categories[j,1]))
    # print(categories_dict)

def create_dataset():

    path = os.path.join(base_dir,'annotations.json')
    print("Reading annotations ... ")
    data = read_annotation(path)
    print("Reading categories ... ")
    cate_dict = read_categories(data)
    print("Reading images ... ")
    image_dict = read_images(data)
    print("Reading new categories ...")
    new_cate_dict = read_csv()
    generate_dataset(data, image_dict, new_cate_dict)

def read_annotation(path):
    # Opening JSON file
    f = open(path, "r")
    
    # returns JSON object as
    # a dictionary
    data = json.load(f)

    return data
def read_categories(data):
    # iterations of the categories in the annotations to get the categories id, name, new id
    cate_dict = {}
    count = 0
    for c in tqdm(data["categories"]):
        cate_dict[c["id"]] = {}
        cate_dict[c["id"]]["name"] = c["name"]
        cate_dict[c["id"]]["cate_id"] = count
        count += 1
    return cate_dict

def read_images(data):
    image_dict = {}

    # iterations for all the images
    for d in tqdm(data["images"]):
        image_dict[d["id"]] = d["file_name"]
    return image_dict

def generate_dataset(data, image_dict, cate_dict):
    for a in tqdm(data["annotations"]):
        filename = image_dict[a["image_id"]]
        name = filename.split('.')[0]
        img = cv2.imread(os.path.join(base_dir,"images",filename))
        for s in a["segmentation"]:
            lines = np.array(s).reshape((1,int(np.array(s).shape[0]/2),2)).astype(np.int32)
            label = None
            try:
                l = int(cate_dict[int(a["category_id"])]["cate_id"])

                if exists(os.path.join(new_dir , name + '.png')):
                    label = cv2.imread(new_dir+'label/'+ name + '.png', cv2.IMREAD_CHANGED)
                else:
                    label = np.zeros((img.shape[0],img.shape[1],1))
                    cv2.imwrite(os.path.join(new_dir,'color', name + '.png'), img)
                
                cv2.fillPoly(label, lines, l)
                cv2.imwrite(os.path.join(new_dir,'label', name + '.png'), label)
            except:
                pass
            
            # cv2.fillPoly(label, lines, int(cate_dict[a["category_id"]]["cate_id"]))
           
            

create_dataset()

# img = cv2.imread("/media/wisccitl/15afc964-cd4b-4320-bb71-364fba832bb1/han/itann_final_project/dataset/val/label/013026.png", cv2.IMREAD_UNCHANGED)
# # img[:,:,2] = 255
# print(img[img[:,:,1] != 0])
# cv2.imshow("Image", img)
# cv2.waitKey()
# cv2.destroyAllWindows()

def create_csv():
    path = os.path.join(base_dir,'annotations.json')
    print("Reading annotations ... ")
    data = read_annotation(path)
    print("Reading categories ... ")
    cate_dict = read_categories(data)
    # print(cate_dict)
    with open('categories.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for key in cate_dict.keys():
            writer.writerow([cate_dict[key]['cate_id'],key,cate_dict[key]['name']])

# create_csv()


# read_csv()