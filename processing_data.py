from numpy.lib.twodim_base import mask_indices
import cv2
import numpy as np
import glob
from tqdm import tqdm
import os
import os.path as osp

# combine result images
directory = "./test_result_whole_img/"
os.makedirs(directory, exist_ok = True)
class_color = [[70,70,70],[150,150,202],[100,186,198],[186,183,167],[133,255,255],[206,192,192],[160,80,32],[1,134,193]]

def prediction_format(img, class_color):
    result = np.zeros((img.shape[0],img.shape[1]))
    for c in range(len(class_color)):
        if c != 0:
            result[(img == class_color[c]).all(2)] = 1
        else:
            result[(img == class_color[c]).all(2)] = 7
    return result

def combine_image():
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
                        # img = prediction_format(img, class_color)
                        cv2.imwrite(osp.join(directory,filename+".png"), img)
        #print(filename)
        #exit(-1)

combine_image()

def colors_for_paper():
	target_names = ['wall','beam','column','window frame','window pane','balcony','slab']
	class_color = [[150,150,202],[100,186,198],[186,183,167],[133,255,255],[206,192,192],[160,80,32],[1,134,193]]
	
	for i in range(len(class_color)):
		img = np.zeros((100,500,3))
		img[:,:]= class_color[i]
		cv2.imwrite(target_names[i]+"_color.png", img)

# colors_for_paper()

def revised_damage_label():
	for file_path in tqdm(glob.glob("/media/wisccitl/15afc964-cd4b-4320-bb71-364fba832bb1/han/data_proj2_QuakeCity/label/crack/*.png")):
		filename = get_filename(file_path)
		img = read_img(file_path)
		img = blur_img(img)
		mask = get_mask(img)
		cv2.imwrite("/media/wisccitl/15afc964-cd4b-4320-bb71-364fba832bb1/han/data_proj2_QuakeCity/label/new_crack/"+filename, mask)
		
def get_filename(file_path):
	filename = file_path.split('/')[-1]
	return filename

def read_img(file_path):
	# print("Read file ", file_path)
	img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
	return img

def blur_img(img):
	# ksize
	ksize = (3, 3)
	# Using cv2.blur() method 
	img = cv2.blur(img, ksize)
	return img

def get_mask(img):
	mask = np.zeros(img.shape)
	mask = np.where((img != [0,0,0]).any(axis=2,  keepdims=True), [0,0,255], mask)
	
	return mask

def show_img(img):
	cv2.imshow("Image", img)
	cv2.waitKey()
	cv2.destroyAllWindows()

# revised_damage_label()

def dataset_smaple():
	image_dir = "/media/wisccitl/15afc964-cd4b-4320-bb71-364fba832bb1/han/data_proj2_QuakeCity/image/"

	# dataset sample for components
	count = 0
	for file in glob.glob("/media/wisccitl/15afc964-cd4b-4320-bb71-364fba832bb1/han/data_proj2_QuakeCity/label/component/*.png"):
		if count %200 == 0:
			filename = file.split("/")[-1]
			print("file: ", filename)
			img = cv2.imread(image_dir + filename)
			label = cv2.imread(file)
			img_component = cv2.addWeighted(img,1.0,label,0.5,0.0)
			cv2.imwrite("/media/wisccitl/15afc964-cd4b-4320-bb71-364fba832bb1/han/data_proj2_QuakeCity/dataset_sample/component/"+filename+"_dataset_sample.png",img_component)
		count += 1
	
	# dataset sample for crack
	spall_label_dir = "/media/wisccitl/15afc964-cd4b-4320-bb71-364fba832bb1/han/data_proj2_QuakeCity/label/spall/"
	rebar_label_dir = "/media/wisccitl/15afc964-cd4b-4320-bb71-364fba832bb1/han/data_proj2_QuakeCity/label/rebar/"
	count = 0
	for file in glob.glob("/media/wisccitl/15afc964-cd4b-4320-bb71-364fba832bb1/han/data_proj2_QuakeCity/label/crack/*.png"):
		if count %200 == 0:
			filename = file.split("/")[-1]
			print("file: ", filename)
			img = cv2.imread(image_dir + filename)
			crack_label = cv2.imread(file)
			spall_label = cv2.imread(spall_label_dir + filename)
			rebar_label = cv2.imread(rebar_label_dir + filename)
			img_crack = cv2.addWeighted(img,1.0,crack_label,0.5,0.0)
			cv2.imwrite("/media/wisccitl/15afc964-cd4b-4320-bb71-364fba832bb1/han/data_proj2_QuakeCity/dataset_sample/crack/"+filename+"_dataset_sample.png",img_crack)
			img_spall = cv2.addWeighted(img,1.0,spall_label,0.5,0.0)
			cv2.imwrite("/media/wisccitl/15afc964-cd4b-4320-bb71-364fba832bb1/han/data_proj2_QuakeCity/dataset_sample/spall/"+filename+"_dataset_sample.png",img_spall)
			img_rebar = cv2.addWeighted(img,1.0,rebar_label,0.5,0.0)
			cv2.imwrite("/media/wisccitl/15afc964-cd4b-4320-bb71-364fba832bb1/han/data_proj2_QuakeCity/dataset_sample/rebar/"+filename+"_dataset_sample.png",img_rebar)
		count += 1

# dataset_smaple()

def dataset_smaple_with_and_without_background():
	image_dir = "/media/wisccitl/15afc964-cd4b-4320-bb71-364fba832bb1/han/data_proj2_QuakeCity/image/"
	count = 0
	background_color = [70,70,70]
	for file in glob.glob("/media/wisccitl/15afc964-cd4b-4320-bb71-364fba832bb1/han/data_proj2_QuakeCity/label/component/*.png"):
		if count %200 == 0:
			filename = file.split("/")[-1]
			print("file: ", filename)
			img = cv2.imread(image_dir + filename)
			label = cv2.imread(file)
			cv2.imwrite("/media/wisccitl/15afc964-cd4b-4320-bb71-364fba832bb1/han/data_proj2_QuakeCity/dataset_sample_with_and_without_background/with_background/"+filename+"_dataset_sample.png",img)
			img[(label == background_color).all(2)] = [0,0,0]
			cv2.imwrite("/media/wisccitl/15afc964-cd4b-4320-bb71-364fba832bb1/han/data_proj2_QuakeCity/dataset_sample_with_and_without_background/without_background/"+filename+"_dataset_sample.png",img)
		count += 1

# dataset_smaple_with_and_without_background()

def dataset_sampe_original_and_revised_crack_label():
	image_dir = "/media/wisccitl/15afc964-cd4b-4320-bb71-364fba832bb1/han/data_proj2_QuakeCity/image/"
	count = 0
	for file_path in tqdm(glob.glob("/media/wisccitl/15afc964-cd4b-4320-bb71-364fba832bb1/han/data_proj2_QuakeCity/label/crack/*.png")):
		if count %200 == 0:
			filename = get_filename(file_path)
			img = cv2.imread(image_dir + filename)
			label = read_img(file_path)
			img_original_crack = cv2.addWeighted(img,1.0,label,0.8,0.0)
			cv2.imwrite("/media/wisccitl/15afc964-cd4b-4320-bb71-364fba832bb1/han/data_proj2_QuakeCity/dataset_sample_original_and_revised_crack_label/original_crack_label/"+filename, img_original_crack)
			
			label = blur_img(label)
			mask = get_mask(label).astype(np.int32)
			img_revised_crack = cv2.addWeighted(img.astype(np.int32),1.0,mask,0.8,0.0)
			cv2.imwrite("/media/wisccitl/15afc964-cd4b-4320-bb71-364fba832bb1/han/data_proj2_QuakeCity/dataset_sample_original_and_revised_crack_label/revised_crack_label/"+filename, img_revised_crack)
		count += 1

# dataset_sampe_original_and_revised_crack_label()

def dataset_sampe_damage_state_label():
	image_dir = "/media/wisccitl/15afc964-cd4b-4320-bb71-364fba832bb1/han/data_proj2_QuakeCity/image/"
	count = 0
	for file_path in tqdm(glob.glob("/media/wisccitl/15afc964-cd4b-4320-bb71-364fba832bb1/han/data_proj2_QuakeCity/label/ds/*.png")):
		if count %200 == 0:
			filename = get_filename(file_path)
			img = cv2.imread(image_dir + filename)
			label = read_img(file_path).astype(np.int32)
			img_with_label = cv2.addWeighted(img.astype(np.int32),1.0,label,0.4,0.0)
			cv2.imwrite("/media/wisccitl/15afc964-cd4b-4320-bb71-364fba832bb1/han/data_proj2_QuakeCity/dataset_sample_damage_state_label/"+filename, img_with_label)
		count += 1

# dataset_sampe_damage_state_label()


		

