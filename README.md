# PredNitro

Post-translational modification (PTM) is defined as the enzymatic changes of proteins after the translation process in protein biosynthesis. 
Nitrotyrosine, which is one of the most important modifications of proteins, is interceded by the active nitrogen molecule. It is known to be 
associated with different diseases including autoimmune diseases characterized by chronic inflammation and cell damage. Currently, Nitrotyrosine 
sites are identified using experimental approaches which are laborious and costly. In this study, we propose a new machine learning method called 
PredNitro to accurately predict Nitrotyrosine sites. To build PredNitro, we use sequence coupling information from the neighboring amino acids of 
tyrosine residues along with a support vector machine as our classification technique.Our results demonstrates that PredNitro achieves 98.0% accuracy
with more than 0.96 MCC and 0.99 AUC in both 5-fold cross-validation and jackknife cross-validation tests which are significantly better than those 
reported in previous studies. PredNitro is publicly available as an online predictor at: 

This repository contains the predictive decision-making workflow of PredNitro. 

## Cross-Validation 
Contains the benchmark dataset and the source code of the M times k-fold cross-validation. 
The M-iterations of k-fold cross-validation were performed according to the following steps:

  Step 1: Divide the extracted dataset randomly into k disjoint sets.

  Step 2: Select 1 set as test set and utilize the remaining k-1 sets as training set.

  Step 3: Train the RBF kernel based SVM predictor with the training set using the optimal hyperparameters (C, gamma) of the respective iteration.

  Step 4: Perform prediction on the test set.

  Step 5: Repeat steps 2 to 4 until all k sets had been used for testing.

  Step 7: Merge the prediction outputs and measure the performance.

  Step 8: Repeat steps 1 to 7 for 10 times.

  Step 9: Measure the average performance of M repetitions with corresponding standard deviations.




## Independent Test 
Contains the independent test dataset and the predicted and test labels with corresponding prediction performances of the respective predictors.
