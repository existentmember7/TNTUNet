import numpy as np
import glob
import cv2
import os
import os.path as osp
from tqdm import tqdm

directory = "./test_result_whole_img/"
os.makedirs(directory, exist_ok = True)

def main():
    filenames = glob.glob("./test_result_img/*.png")
    file_dict = []
    for count in tqdm(range(len(filenames))):
        filename = filenames[count].split('/')[-1].split('_')[0]
        for f in filenames:
            if filename not in file_dict:
                if filenames[count] != f:
                    temp_filename = f.split('/')[-1].split('_')[0]
                    if temp_filename == filename:
                        img_l = None
                        img_r = None
                        img = np.zeros((1080,1920,3))
                        if f.split("/")[-1].split('.')[0].split('_')[-1] == 'l':
                            img_l = cv2.imread(f)
                            img_r = cv2.imread(filenames[count])
                        else:
                            img_l = cv2.imread(filenames[count])
                            img_r = cv2.imread(f)
                        img_l = cv2.resize(img_l, (1080, 1080), interpolation = cv2.INTER_AREA)
                        img_r = cv2.resize(img_r,(1080,1080), interpolation = cv2.INTER_AREA)
                        img[:,:1080] = img_l
                        img[:,1920-1080:] = img_r
                        file_dict.append(filename)
                        cv2.imwrite(osp.join(directory,filename+".png"), img)
        #print(filename)
        #exit(-1)

main()
