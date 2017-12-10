## Paper Link
[Colorimetric Detection of pH Strips](paper/paper.pdf)

## Code Descriptions
ml_pipeline contains the code to do grid search on various machine learning pipelines. To add a new custom pipeline, you can add it to pipelines.py, add its parameters to parameters.py, and then run it in ModelGridSearch.ipynb

preprocessing_util.py contains functions to process raw images, including masking, color extraction, and recoloring such as gamma modification and color rotation. 

util.py contains a function that will load all images and apply the desired recoloring.