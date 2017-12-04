## Repo Structure
Final deliverables intended for BioE134 can be found in the BioE134 directory. Final deliverables intended for CS189 can be found in the CS189 directory. Common files used for both projects remain in the root directory.

## Authors
#### Rudra Mehta, Matthew Sit, Aman Dhar.
All three of us are undergraduates in CS 189, completing this project for extra credit for Fall 2017. Rudra and Matthew intend on submitting a more biological view on this topic for their final project for BioE 134, Genetic Design Automation. Some background research, data collection, and fundamental modeling approaches may overlap, but the written deliverables and area of analyses will be unique. Permission to submit a related project to 189 has been provided by the BioE 134 professor, John Christopher Anderson (jcanderson [at] berkeley.edu), as well as CS 189 professor, Anant Sahai (sahai [at] eecs.berkeley.edu).

## Project Overview
The project is to create and test various machine learning models to conclude upon the best way to find the pH of a biochemical solution, given an image of a pH test strip that has been applied to that solution.

## Background
Growing cell cultures for industrial-scale chemical output is a very difficult task. Large bioreactors cost tens of thousands of dollars and require complex, specialized machinery to operate. An alternative idea is to make use of many 5-gallon buckets, with a film on top to allows air to pass through the system, but blocks liquids. With a pump inside the bucket to mix nutrients with the cells, we essentially have a low-cost bioreactor. We need to test the pH of each bucket periodically to determine when to feed the cells. This can be done in a cost-effective manner by using paper-based strips that are briefly submerged into the solution. This method of bioreactors can only be profitable if there are many buckets - on the order of hundreds or thousands. Therefore, we need an efficient, automated method of determining pH from the test strips. The literature on this suggest that in practice, smartphones are used as the source of these pictures. This poses an extra challenge because this means that such algorithms must be robust in handling various lighting and positional arrangement conditions.

## Proposal
We propose a machine learning model to calculate the numerical pH value from an image of each of these strips. We will try several pre-processing techniques, including color variation stabilization via a printed calibration chart (as in Kim, et al), a color histogram, and normalization. We may also use canonical correlation analysis to project images of the test strips into lower-dimensional subspaces (as done in HW10, #4) to aid in feature selection and computation. To classify the images, we will try several models with hyper-parameter tuning, including Ridge-Regularized Least Squares, Neural Nets, and Least Squares Support Vector Machines (as in Mutlu, et al). Ultimately, we will exhaustively test several combinations of pre-processing techniques, classifiers, and hyperparameters to settle on the combination that achieves the highest accuracy on a test set.

If we are able to successfully reach a classifier that works best, we may try to integrate the code into a simple, lightweight iOS app (as explored in Mutlu, et al). This would potentially help researchers classify test strips much more quickly. In order to achieve this, we would likely have to translate our python code into Swift, and use the Upsurge library to implement most of the mathematical calculations. We may also try to compile our python code on an Arduino, which in theory would allow robotic devices to both run experiments with the test strips and automatically classify each strip without human assistance.

## References
 - A Smartphone-Based Automatic Measurement Method for Colorimetric pH Detection Using a Color Adaptation Algorithm, Kim, et al. (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5539506/).
 - Smartphone Based Colorimetric Detection via Machine Learning, Mutlu, et al. (https://www.researchgate.net/profile/Mehmet_Solmaz3/publication/315710086_Smartphone_Based_Colorimetric_Detection_via_Machine_Learning/links/59dcb9810f7e9b14600468e5/Smartphone-Based-Colorimetric-Detection-via-Machine-Learning.pdf).
 - Upsurge, math utilities library for Swift.
