import sys
import argparse

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.utils import shuffle

from pipelines import *
from parameters import *

model_job = 'test'
job_type = 'regression'
images_path = '../../data_matrices/'

def get_train_data(images_path, job_type):
    # convert from images to np arrays
    X = np.load(images_path + "X.npy")
    Y = np.load(images_path + "Y.npy")
    if job_type == 'classification':
        Y = np.array([int(np.round(y)) for y in Y])
    return X, Y

def get_pred_data(images_path):
    # convert from images to np arrays
    X = np.load(images_path + "X.npy")
    return X

def load_possible_models(job_type):
    model_dict = {}
    if job_type == 'regression':
        model_dict['ridge'] = (ridge_regression_pipeline, ridge_regression_parameters)
        model_dict['lasso'] = (lasso_regression_pipeline, lasso_regression_parameters)
        model_dict['en'] = (elastic_net_regression_pipeline, elastic_net_regression_parameters)
        model_dict['knn'] = (knn_regression_pipeline, knn_regression_parameters)
    elif job_type == 'classification':
        model_dict['knn'] = (knn_classification_pipeline, knn_classification_parameters)
        model_dict['svm'] = (svm_classification_pipeline, svm_classification_parameters)
        model_dict['lda'] = (lda_classification_pipeline, lda_classification_parameters)
        model_dict['qda'] = (qda_classification_pipeline, qda_classification_parameters)
    return model_dict

def grid_search(X, Y, pipeline, parameters, job_type):
    if job_type == 'regression':
        scoring_fn = 'neg_mean_squared_error'
    elif job_type == 'classification':
        scoring_fn = 'accuracy'
    grid = GridSearchCV(estimator=pipeline, param_grid=parameters, scoring=scoring_fn)
    grid.fit(X, Y)
    return grid.best_score_, grid.best_estimator_

def main():
    parser = argparse.ArgumentParser(description='Work on a model.')
    parser.add_argument('--model_job', type=str, help='train or test')
    parser.add_argument('--job_type', type=str, help='regression or classification')
    parser.add_argument('--model_path', type=str, help='path to load the model from or save the model to')
    parser.add_argument('--images_path', type=str, help='path to the images')
    args = parser.parse_args()

    model_job = args.model_job
    job_type = args.job_type
    model_path = args.model_path
    images_path = args.images_path

    if model_job == 'train':
        X, Y = get_train_data(images_path, job_type)
        model_dict = load_possible_models(job_type)
        best_score = -10000000
        best_model = None
        for model_name in model_dict:
            model_pipeline, model_parameters = model_dict[model_name]
            score, model = grid_search(X, Y, model_pipeline, model_parameters, job_type)
            print(score)
            print('\n')
            print(model)
            print('\n\n')
            if score > best_score:
                best_score = score
                best_model = model
        if job_type == 'regression':
            best_score = -best_score
        print(best_score)
        print('\n\n')
        print(best_model)
        joblib.dump(best_model, model_path, compress = 1)

    if model_job == 'test':
        try:
            model = joblib.load(model_path)
        except FileNotFoundError as e:
            print("No trained model found.")
            sys.exit()

        X = get_pred_data(images_path)
        preds = model.predict(X)
        out_file = open('pred_vals.txt', 'w+')

        for pred in preds:
            out_file.write(str(pred) + '\n')
        out_file.close()

if __name__ == '__main__':
    main()
