# coding: utf-8

import numpy as np
import cv2
import numpy.ma as ma
import cv2
from matplotlib import pyplot as plt
from skimage import filters
from scipy import ndimage
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


def convert_image_to_data(image):
    masked_image = create_masked_image(image)
    return get_data_from_masked_image(masked_image)


def partition_image(img_masked, num_partitions = 4):
    zero_row = np.zeros(img_masked.shape[1:])
    partitions = []
    cut_start_index = 0
    should_cut = False
    
    for i, row in enumerate(img_masked):
        row_is_zero = np.all(np.equal(row, zero_row))
        if not should_cut and not row_is_zero:
            should_cut = True
        elif should_cut and row_is_zero:
            partitions.append(img_masked[cut_start_index : i + 1])
            cut_start_index = i + 1
            should_cut = False
            if len(partitions) == num_partitions:
                return partitions
    print("Warning: Didn't reach full number of partitions {}, \
        returning {} partitions instead".format(num_partitions, len(partitions)))
    return partitions


def compute_avg_colored_pixel(partition):
    zero_row = np.zeros(partition.shape[2], dtype='int64')
    pixels = partition.reshape(-1, partition.shape[-1])
    non_black_pixel_indices = np.where(np.any(pixels != zero_row, axis=1))
    non_black_pixels = pixels[non_black_pixel_indices]
    return np.average(non_black_pixels, axis=0)


def get_data_from_masked_image(masked_image):
    partitions = partition_image(masked_image)
    avg_pixels = [compute_avg_colored_pixel(partition) for partition in partitions]
    X = np.array(avg_pixels).flatten()
    return X


def create_masked_image(img):
    img = img[5:-500,25:-25]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    kernel = np.ones((3,3),np.uint8)
    init_closed = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel, iterations = 10)

    gray = init_closed
    sobel_v = filters.sobel_v(gray)
    sobel_h = filters.sobel_h(gray)
    sobel_v = filters.gaussian(sobel_v, sigma=2.0)
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
    while num_blobs < 4:
        opening = cv2.morphologyEx(closing,cv2.MORPH_OPEN,kernel, iterations = iters)
        iters += 5
        blobs, num_blobs = ndimage.label(opening)

    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.4*dist_transform.max(),255,0)

    mask = cv2.cvtColor(sure_fg,cv2.COLOR_GRAY2RGB)
    img_masked = cropped_img.copy()
    img_masked[mask==0] = 0
    return img_masked


def adjust_gamma(image, gamma=None):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    if gamma is None:
        gamma = np.random.uniform(0.5, 1.5)
        # gamma = 0.75
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype('uint8')

    # apply gamma correction using the lookup table
    return cv2.LUT(image.astype('uint8'), table)
    

def rotate_colors(img, degrees=None):

    if degrees is None:
        degrees = np.random.uniform(-90, 90)

    hsv_img = rgb_to_hsv(img/255)
    hsv_img[:, :, 0] = (hsv_img[:, :, 0] + -degrees/360.0) % 1.0
    rgb_img = hsv_to_rgb(hsv_img)
    rgb_img *= 255

    return rgb_img


def recolor_image(image, color_option):
    img_cpy = image.copy()
    if color_option == 'gamma':
        img_cpy = adjust_gamma(img_cpy)
    elif color_option == 'rotate':
        img_cpy = rotate_colors(img_cpy)
    elif color_option == 'mixed':
        img_cpy = adjust_gamma(img_cpy)
        img_cpy = rotate_colors(img_cpy)
    return img_cpy

