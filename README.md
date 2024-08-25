# CodeAlpha_Task1_Handwritten-Character-Recognition

# Disease Prediction using Machine Learning

This repository contains a complete machine learning pipeline for predicting diseases based on symptoms using the Disease-Symptom dataset from Kaggle. The project involves data preprocessing, model building, and advanced visualization techniques to create a robust and interpretable disease prediction model.

## Project Overview

### 1. **Data Exploration and Preprocessing**
   - Loaded and explored the Disease-Symptom dataset, checking for missing values and statistical insights.
   - Preprocessed the data by handling missing values, scaling features, and splitting the data into training and testing sets.

### 2. **Model Building**
   - Built a RandomForestClassifier to predict diseases based on the provided symptoms.
   - Evaluated the model using classification metrics like precision, recall, F1-score, and a confusion matrix.

### 3. **Advanced Visualization**
   - **Feature Importance Plot:** Visualized the importance of each feature in making predictions.
   - **PCA Visualization:** Applied Principal Component Analysis to reduce the dimensionality of the data and visualize it in a 2D space.
   - **ROC Curve:** Plotted the Receiver Operating Characteristic curve to evaluate model performance.
   - **SHAP Values:** Used SHAP (SHapley Additive exPlanations) to explain individual predictions and visualize the impact of each feature.

### 4. **Model Accuracy**
   - Calculated the accuracy of the model on the test dataset, achieving a high level of accuracy in predicting diseases.

## Installation and Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/disease-prediction.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook or script to see the results and visualizations.

## Dataset

The dataset used in this project is the Disease-Symptom dataset, which can be found [here](https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset).

## Results and Insights

The model performed exceptionally well with an accuracy of `X.XX%`. Advanced visualizations like feature importance plots, PCA, ROC curves, and SHAP values provided deep insights into the model's decision-making process, ensuring transparency and interpretability.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, feel free to submit a pull request or open an issue.

