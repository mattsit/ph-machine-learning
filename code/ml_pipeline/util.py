import numpy as np
from sklearn.utils import shuffle
import cv2
import glob

from preprocessing_utils import create_masked_image
from preprocessing_utils import recolor_image
from preprocessing_utils import get_data_from_masked_image



def load_data(pipeline_type, raw=True, gamma=False, rotate=False, mixed=False):

    X_data = []
    Y_data = []
    for img_path in glob.glob('../../raw_data/*.jpg'):
        img = cv2.imread(img_path)
        y = float(img_path[-11:-7].replace('p', '.'))

        raw_masked = create_masked_image(img)

        if raw:
            a = get_data_from_masked_image(raw_masked)
            X_data.append(get_data_from_masked_image(raw_masked))
            Y_data.append(y)
        if gamma:
            gamma_masked = recolor_image(raw_masked, 'gamma')
            X_data.append(get_data_from_masked_image(gamma_masked))
            Y_data.append(y)
        if rotate:
            rotate_masked = recolor_image(raw_masked, 'rotate')
            X_data.append(get_data_from_masked_image(rotate_masked))
            Y_data.append(y)
        if mixed:
            mixed_masked = recolor_image(raw_masked, 'mixed')
            X_data.append(get_data_from_masked_image(mixed_masked))
            Y_data.append(y)

    X_data = np.asarray(X_data)
    
    if pipeline_type == 'classification': 
        Y_data = np.array([int(np.round(y)) for y in Y_data])
    else:
        Y_data = np.array(Y_data).reshape(-1,1) # makes shape (90, 1)

    return shuffle(X_data, Y_data)