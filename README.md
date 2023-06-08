# Credit Risk Analysis Report

## Overview of the Analysis

### Description

This analysis presents the results of my investigation into credit risk analysis using various machine learning techniques. The goal was to train and evaluate Logistic Regression Models to determine the creditworthiness of borrowers. I employed different methods to train the models and compared their performances to identify the superior model. In this analysis, the predictive variables in the model are categorized as 0 (healthy loan) and 1 (high-risk loan).

During the construction of the models, I divided the dataset into features and labels, and further split them into training and testing sets. Here are the details of the two machine learning models used:

Machine Learning Model 1: I created this model by instantiating a logistic regression model and training it with the original training sets (X_train, y_train). I fitted the model to the training sets and used it to generate predictions.
Machine Learning Model 2: For this model, I resampled the original training data using the RandomOverSampler module. Then, I instantiated a logistic regression model and fitted it to the resampled training sets (X_resample, y_resample). Finally, I generated predictions using this model.
To evaluate the performance of each model, I considered metrics such as the balance accuracy score, the confusion matrix, as well as the precision score, recall score, and f1-score from the classification report.

### Results

Machine Learning Model 1:
Model 1 - trained on the original data, achieved an accuracy of 94.4% in predicting the two labels. This model exhibited excellent performance in predicting healthy loans, with both precision and recall scores of 1.00. However, there is room for improvement in predicting high-risk loans. The precision score for high-risk loans was 0.87, indicating that only 87% of the actual high-risk loans were correctly predicted. The recall score for high-risk loans was 0.89, indicating that the model only identified 89% of all high-risk loans in the dataset.
Machine Learning Model 2:
Model 2 - trained on the resampled data, achieved an accuracy of 99.6% in predicting the two labels. This model demonstrated strong performance in predicting healthy loans, with both precision and recall scores of 1.00. The precision score for high-risk loans remained at 0.87, but the recall score improved to 1.00, indicating that the model can now predict all high-risk loans in the dataset.

### Summary

Based on my analysis, it is evident that Model 2 outperforms Model 1 in predicting high-risk loans and exhibits an overall higher accuracy in predicting both labels. Specifically, Model 2 achieved a relatively high precision in predicting high-risk loans while correctly identifying all high-risk loans in the dataset. This performance is considered relatively good in this context. Therefore, I recommend using Model 2 for identifying high-risk loans and achieving better overall accuracy in predicting labels.
