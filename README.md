## Abstract
In this project, we apply several machine learning
techniques and models for the purpose of classifying and estimating
the pH of solutions given raw image data of pH test
strips. We replicate and critique much of the work performed by
Mutlu et. al, who used LS-SVM and claimed to achieve 100%
classification. We believe that this high accuracy was the result
of duplicating physical pH strips samples between training and
validation datasets; this is problematic because the noise between
different strips of the same class was not accounted for, and
samples were essentially duplicated as pre-processing mitigated
orientation variability efforts. We further find that regression is
a more suitable approach for this domain, as pH values are on
a continuous, logarithmic scale, and decimal differences in value
can have significant biological consequences. In this spirit, we
find that mean squared regression errors as low as ~0.033 are
achievable.

## Authors
#### Aman Dhar, Rudra Mehta, Matthew Sit.
All three of us are undergraduates in CS 189, completing this project for extra credit for Fall 2017. Rudra and Matthew submitted a more biological view on this topic for their final project for BioE 134, Genetic Design Automation. Some background research, data collection, and fundamental modeling approaches may overlap, but the written deliverables and area of analyses will be unique. Permission to submit a related project to 189 has been provided by the BioE 134 professor, John Christopher Anderson (jcanderson [at] berkeley.edu), as well as CS 189 professor, Anant Sahai (sahai [at] eecs.berkeley.edu).

## References
1. J. Anderson, 2017 10 09-Final Project, UC Berkeley, 2017.
2. A. Mutlu, V. Kl, G. zdemir, A. Bayram, N. Horzum and M. Solmaz, Smartphone-based colorimetric detection via machine learning, The Analyst, vol. 142, no. 13, pp. 2434-2441, 2017.
3. S. Kim, Y. Koo and Y. Yun, A Smartphone-Based Automatic Measurement Method for Colorimetric pH Detection Using a Color Adaptation Algorithm, Sensors, vol. 17, no. 7, p. 1604, 2017.
4. H. Farid, Blind inverse gamma correction, IEEE Transactions on Image Processing, vol. 10, no. 10, pp. 1428-1433, 2001.
5. American Type Culture Collection, C2C12 (ATCC®CRL-177TM), Product Sheet, Manassas, VA.
