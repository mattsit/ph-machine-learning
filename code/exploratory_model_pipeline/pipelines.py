from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


ridge_regression_pipeline = Pipeline(
        [

            # Apply PCA to ridge
            # ('pca', PCA()),

            # Apply scaling to Ridge Regression
            ('scale', StandardScaler()),

            ('ridge', Ridge())
        ]
    )

lasso_regression_pipeline = Pipeline(
        [

            # Apply PCA to LASSO
            ('pca', PCA()),

            # Apply scaling to Lasso Regression
            ('scale', StandardScaler()),

            
            ('lasso', Lasso())
        ]
    )

elastic_net_regression_pipeline = Pipeline(
        [

            # Apply PCA
            ('pca', PCA()),

            # Apply scaling
            ('scale', StandardScaler()),
            
            ('en', ElasticNet())
        ]
    )




knn_regression_pipeline = Pipeline(
        [
            ('knn', KNeighborsRegressor())
        ]
    )


svm_classification_pipeline = Pipeline(
        [
            # Apply PCA to SVM Classification
            ('pca', PCA()),

            # Apply scaling to SVM Classification
            ('scale', StandardScaler()),

            ('svm', SVC())
        ]
    )


knn_classification_pipeline = Pipeline(
        [
            ('knn', KNeighborsClassifier())
        ]
    )


lda_classification_pipeline = Pipeline(
        [
            ('pca', PCA()),
            ('scale', StandardScaler()),

            ('lda', LinearDiscriminantAnalysis())
        ]
    )

qda_classification_pipeline = Pipeline(
        [
            ('pca', PCA()),
            ('scale', StandardScaler()),

            ('qda', QuadraticDiscriminantAnalysis())
        ]
    )


