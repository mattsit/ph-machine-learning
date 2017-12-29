## Paper
[Colorimetric Detection of pH Strips](paper/paper.pdf)

## File Descriptions
exploratory_model_pipeline and exploratory_ridge_on_different_amounts_of_preprocessing contain code from preliminary tests on a wider variety of models.

figures contains plots and images used in the paper as well as those generated but not used.

ml_pipeline contains the code to do grid search on various machine learning pipelines. To add a new custom pipeline, you can add it to pipelines.py, add its parameters to parameters.py, and then run it in ModelGridSearch.ipynb

paper contains the actual report itself and the helper files used to generate it.

preprocessing contains various scripts used to process our images prior to exposing them to our machine learning pipeline. In particular, preprocessing_util.py contains functions to process raw images, including masking, color extraction, and recoloring such as gamma modification and color rotation. Further, util.py contains a function that will load all images and apply the desired recoloring.

demo_presentation.pdf is a slide deck explaining the results of our exploratory preliminary tests.

(Note that some file paths may no longer be functional since this repo has been re-structured for better readability.)



