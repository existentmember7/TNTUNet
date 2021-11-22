import cv2
import numpy as np

img = cv2.imread("/media/wisccitl/15afc964-cd4b-4320-bb71-364fba832bb1/han/data_proj2_QuakeCity/label/component/A10001.png")
unique = np.unique(img.reshape(-1, img.shape[2]), axis=0)
print(unique)
