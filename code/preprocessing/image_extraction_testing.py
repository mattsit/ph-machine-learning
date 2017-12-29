import numpy as np
import numpy.ma as ma
import cv2
from matplotlib import pyplot as plt
import os

from skimage import filters

from scipy import ndimage

def save_color_im(img, f_name):
    plt.imsave(masked_data_dir + f_name, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

raw_data_dir = "raw_data/"
masked_data_dir = "masked_data/"


for in_file in os.listdir(raw_data_dir):
    if in_file[-3:] != "jpg":
        continue
    print(in_file)


    img = cv2.imread(raw_data_dir + in_file)
    img = img[5:-500,25:-25]
    gray_raw = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


    kernel = np.ones((3,3),np.uint8)
    gray = cv2.morphologyEx(gray_raw,cv2.MORPH_CLOSE,kernel, iterations = 10)

    sobel_v = filters.sobel_v(gray)
    sobel_v = filters.gaussian(sobel_v, sigma=2.0)
    sobel_h = filters.sobel_h(gray)
    sobel_h = filters.gaussian(sobel_h, sigma=2.0)
    mask_v = np.abs(sobel_v) > .03
    mask_h = np.abs(sobel_h) > .03

    # Coordinates of non-black pixels.
    coords_y = np.argwhere(mask_v)
    coords_x = np.argwhere(mask_h)

    # Bounding box of non-black pixels.
    x0 = max(coords_x.min(axis=0)[0] - 50,0)
    x1 = coords_x.max(axis=0)[0] + 51
    y0 = max(coords_y.min(axis=0)[1] - 50,0)
    y1 = coords_y.max(axis=0)[1] + 51

    # Get the contents of the bounding box.
    cropped_gray = gray[x0:x1, y0:y1]
    cropped_img = img[x0:x1, y0:y1]

    thresh_val = np.median(cropped_gray)/1.05
    ret, thresh = cv2.threshold(cropped_gray,thresh_val,255,cv2.THRESH_BINARY_INV)


    kernel = np.ones((3,3),np.uint8)
    closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 2)
    blobs, num_blobs = ndimage.label(closing)
    iters = 5
    opening = closing
    while num_blobs != 4:
        opening = cv2.morphologyEx(closing,cv2.MORPH_OPEN,kernel, iterations = iters)
        iters += 5
        blobs, num_blobs = ndimage.label(opening)




    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.4*dist_transform.max(),255,0)




    mask = cv2.cvtColor(sure_fg,cv2.COLOR_GRAY2RGB)
    img_masked = cropped_img.copy()
    img_masked[mask==0] = 0
    save_color_im(img_masked, in_file)

