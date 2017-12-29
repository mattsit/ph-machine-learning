import numpy as np

ridge_regression_parameters = {

    # Hyperparameter search over all possible dimensions for PCA reduction
    'pca__n_components': np.arange(1, 13),

    # 'ridge__alpha': np.arange(0, .05, .001)
    'ridge__alpha': [.001,.01,.1,1,10]
}

lasso_regression_parameters = {
    
    # Hyperparameter search over all possible dimensions for PCA reduction
    'pca__n_components': np.arange(1, 13),

    'lasso__alpha': [.001,.01,.1]#np.arange(0, 0.1, 0.01)
}

elastic_net_regression_parameters = {
    
    # Hyperparameter search over all possible dimensions for PCA reduction
    # 'pca__n_components': np.arange(1, 13),

    'en__alpha': np.arange(.001, .1, .01),

    'en__l1_ratio': np.arange(0, 3, 0.1)
}

knn_regression_parameters = {

    # Number of neighbors to use
    'pca__n_components': np.arange(1, 13),
    'knn__n_neighbors': np.arange(1, 16),


    # Apply weightings vs k for k Nearest Neighbors Regression
    'knn__weights': ['uniform','distance']
}

svm_classification_parameters = {

    # Hyperparameter search over all possible dimensions for PCA reduction
    'pca__n_components': np.arange(1, 13),

    # SVM kernel type
    'svm__kernel': ['linear'],

    # C hyperparameter, original
    'svm__C': np.arange(.001,1.02,.25),
}

knn_classification_parameters = {

    # Number of neighbors to use
    # 'pca__n_components': np.arange(1, 13),
    'knn__n_neighbors': np.arange(1, 16),

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


