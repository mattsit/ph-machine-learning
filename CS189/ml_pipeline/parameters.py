import numpy as np

ridge_regression_parameters = {

    # Hyperparameter search over all possible dimensions for PCA reduction
    # 'pca__n_components': np.arange(1, 13),

    # 'ridge__alpha': np.arange(0, .05, .001)
    'ridge__alpha': np.arange(.5, 2, .1)
}

lasso_regression_parameters = {
    
    # Hyperparameter search over all possible dimensions for PCA reduction
    'pca__n_components': np.arange(1, 13),

    'lasso__alpha': np.arange(0.02, 0.1, 0.001)
}

elastic_net_regression_parameters = {
    
    # Hyperparameter search over all possible dimensions for PCA reduction
    'pca__n_components': np.arange(1, 13),

    'en__alpha': np.arange(.01, .05, .001),

    'en__l1_ratio': np.arange(2.5, 3, 0.01)
}

knn_regression_parameters = {

    # Number of neighbors to use
    'knn__n_neighbors': np.arange(1, 20),


    # Apply weightings vs k for k Nearest Neighbors Regression
    'knn__weights': ['uniform','distance']
}

svm_classification_parameters = {

    # Hyperparameter search over all possible dimensions for PCA reduction
    'pca__n_components': np.arange(1, 13),

    # SVM kernel type
    'svm__kernel': ['linear','rbf'],

    # C hyperparameter, original
    'svm__C': np.arange(.1, 8, .5),

    # Gamma hyperparameter
    'svm__gamma': np.arange(.01, .2, .1)
}

knn_classification_parameters = {

    # Number of neighbors to use
    'knn__n_neighbors': np.arange(1, 30),

    # Apply weightings vs k for k Nearest Neighbors Classification
    'knn__weights': ['uniform', 'distance']
}

lda_classification_parameters = {

    # Hyperparameter search over all possible dimensions for PCA reduction
    'pca__n_components': np.arange(1, 13),

    # Shrinkage to use, if the solver is not 'svd'
    # 'lda__shrinkage': np.arange(.01, 1, .1),

    # Type of solver to use
    # 'lda__solver': ['eigen', 'lsqr']
    'lda__solver': ['svd']
}

qda_classification_parameters = {

    # Hyperparameter search over all possible dimensions for PCA reduction
    'pca__n_components': [7],#np.arange(1, 13),

    # Regularization parameter
    'qda__reg_param': np.arange(.01, .4, .02)
}


