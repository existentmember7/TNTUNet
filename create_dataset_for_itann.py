import numpy as np
import json
import cv2
from os.path import exists
from tqdm import tqdm
import csv

dataset_type = "train"
base_dir = '/media/wisccitl/15afc964-cd4b-4320-bb71-364fba832bb1/han/itann_final_project/train-v0.4/'+dataset_type+'/'
new_dir = '/media/wisccitl/15afc964-cd4b-4320-bb71-364fba832bb1/han/itann_final_project/dataset/'+dataset_type+'/'

def create_dataset():

    path = base_dir+'annotations.json'
    print("Reading annotations ... ")
    data = read_annotation(path)
    print("Reading categories ... ")
    cate_dict = read_categories(data)
    print("Reading images ... ")
    image_dict = read_images(data)
    generate_dataset(data, image_dict, cate_dict)

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
        img = cv2.imread(base_dir+"images/"+filename)
        for s in a["segmentation"]:
            lines = np.array(s).reshape((1,int(np.array(s).shape[0]/2),2)).astype(np.int32)
            label = None
            if exists(new_dir + name + '.png'):
                label = cv2.imread(new_dir+'label/'+ name + '.png', cv2.IMREAD_CHANGED)
            else:
                label = np.zeros((img.shape[0],img.shape[1],3))
                cv2.imwrite(new_dir+'color/'+ name + '.png', img)
            # cv2.fillPoly(label, lines, int(cate_dict[a["category_id"]]["cate_id"]))
            l = int(cate_dict[a["category_id"]]["cate_id"])
            l_0 = (l//255)*255 if l>=255 else l
            l_1 = l%255 if l>=255 else 0
            l_2 = 0
            cv2.fillPoly(label, lines, (l_0,l_1,l_2))
            cv2.imwrite(new_dir+'label/'+ name + '.png', label)
            # if name == "013026":
            #     print(((l//255)*255,l%255,0))
            #     exit(-1)
            

# create_dataset()

# img = cv2.imread("/media/wisccitl/15afc964-cd4b-4320-bb71-364fba832bb1/han/itann_final_project/dataset/val/label/013026.png", cv2.IMREAD_UNCHANGED)
# # img[:,:,2] = 255
# print(img[img[:,:,1] != 0])
# cv2.imshow("Image", img)
# cv2.waitKey()
# cv2.destroyAllWindows()

def create_csv():
    path = base_dir+'annotations.json'
    print("Reading annotations ... ")
    data = read_annotation(path)
    print("Reading categories ... ")
    cate_dict = read_categories(data)
    # print(cate_dict)
    with open('categories.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for key in cate_dict.keys():
            writer.writerow([cate_dict[key]['cate_id'],key,cate_dict[key]['name']])

create_csv()