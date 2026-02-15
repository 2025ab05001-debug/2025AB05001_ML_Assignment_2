1. Problem Statement
The objective of this project is to build and evaluate multiple machine learning classification models on a chosen dataset and deploy the trained models in an interactive web application using Streamlit. The goal is to compare the performance of different classifiers using standard evaluation metrics and provide a simple interface for testing the models on new data.


2. Dataset Description

The dataset used in this project is a classification dataset (Heart Disease dataset from UCI/Kaggle). It contains patient health-related features such as age, sex, chest pain type, resting blood pressure, cholesterol levels, maximum heart rate, and other clinical parameters. The target variable represents the presence or absence of heart disease. The dataset contains more than 500 instances and more than 12 features, making it suitable for training and evaluating multiple classification models.

3. Models Used and Evaluation Metrics

The following machine learning models were implemented and evaluated on the same dataset:

| Model               | Accuracy | AUC  | Precision | Recall | F1 Score | MCC  |
| ------------------- | -------- | ---- | --------- | ------ | -------- | ---- |
| Logistic Regression | 0.84     | 0.93 | 0.82      | 0.89   | 0.85     | 0.69 |
| Decision Tree       | 0.98     | 0.99 | 1.00      | 0.97   | 0.99     | 0.97 |
| KNN                 | 0.70     | 0.83 | 0.71      | 0.69   | 0.70     | 0.40 |
| Naive Bayes         | 0.84     | 0.91 | 0.83      | 0.88   | 0.85     | 0.69 |
| Random Forest       | 1.00     | 1.00 | 1.00      | 1.00   | 1.00     | 1.00 |
| XGBoost             | 1.00     | 1.00 | 1.00      | 1.00   | 1.00     | 1.00 |

4. Observations on Model Performance

Logistic Regression:
Logistic Regression performed well as a baseline model with good overall accuracy and a balanced precision–recall trade-off. However, since it is a linear model, it is limited in capturing complex non-linear relationships present in the dataset.

Decision Tree:
The Decision Tree model achieved very high performance, showing its ability to capture non-linear patterns in the data. However, decision trees are prone to overfitting, especially when the tree becomes deep and fits noise in the training data.

K-Nearest Neighbors (KNN):
KNN showed comparatively lower performance than other models. Its results are sensitive to feature scaling and the choice of the number of neighbors (k), which can significantly impact prediction quality.

Naive Bayes:
Naive Bayes provided stable and reasonably good performance with simple probabilistic assumptions. It works well for classification tasks but may struggle when features are highly correlated, which can affect prediction accuracy.

Random Forest:
Random Forest achieved the best overall performance across all evaluation metrics. The ensemble nature of the model helps in reducing overfitting and improves generalization compared to a single decision tree.

XGBoost:
XGBoost delivered excellent performance with strong predictive power due to gradient boosting. While it provides high accuracy and robustness, it is more complex and computationally intensive compared to simpler models.
|
