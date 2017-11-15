import numpy as np
import numpy.ma as ma
import cv2
from matplotlib import pyplot as plt
import os

def save_color_im(img, f_name):
    plt.imsave(masked_data_dir + f_name, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

raw_data_dir = "raw_data/"
masked_data_dir = "masked_data/"


for in_file in os.listdir(raw_data_dir):
    if in_file[-3:] != "jpg":
        continue

    img = cv2.imread(raw_data_dir + in_file)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    thresh_val = np.median(gray)/1.15
    ret, thresh = cv2.threshold(gray,thresh_val,255,cv2.THRESH_BINARY_INV)

    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    closing = cv2.morphologyEx(opening,cv2.MORPH_CLOSE,kernel, iterations = 2)

    dist_transform = cv2.distanceTransform(closing,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)

    mask = cv2.cvtColor(sure_fg,cv2.COLOR_GRAY2RGB)
    img_masked = img.copy()
    img_masked[mask==0] = 0
    save_color_im(img_masked, in_file)

