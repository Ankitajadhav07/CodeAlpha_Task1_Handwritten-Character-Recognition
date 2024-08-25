# CodeAlpha_Task1_Handwritten-Character-Recognition

# Handwritten Character Recognition

This repository contains a complete implementation of a handwritten character recognition system using a Convolutional Neural Network (CNN). The project combines the A-Z Handwritten Alphabets dataset with the MNIST digits (1-9) to build a model that recognizes both digits and uppercase English letters.

## Project Overview

The objective of this project is to develop a machine learning model that can accurately recognize handwritten characters, including digits (1-9) and uppercase letters (A-Z). The project involves the following steps:

1. **Data Preparation**:
   - **A-Z Handwritten Alphabets Dataset**: A dataset of handwritten uppercase English alphabets (A-Z).
   - **MNIST Dataset**: A dataset of handwritten digits (0-9), from which only digits 1-9 are used.
   - **Combining the Datasets**: The A-Z dataset is combined with the MNIST dataset to create a dataset with 35 classes (1-9, A-Z).

2. **Model Development**:
   - **Convolutional Neural Network (CNN)**: A CNN model is built to learn features from the combined dataset.
   - **Training**: The model is trained on the combined dataset with multiple epochs, using `Adam` optimizer and `sparse_categorical_crossentropy` loss function.

3. **Evaluation**:
   - **Test Accuracy**: The model is evaluated on the test set, achieving competitive accuracy.
   - **Confusion Matrix**: A confusion matrix is plotted to visualize the model's performance across different classes.

4. **Visualization**:
   - **Sample Visualizations**: Example images from both datasets are visualized.
   - **Misclassified Images**: Some of the misclassified images are plotted to understand where the model struggles.
   - **Accuracy and Loss Plots**: Training and validation accuracy and loss curves are plotted to monitor the model’s learning process.

5. **Model Saving**:
   - The trained model is saved for future inference.

## Interesting Plots

- **Confusion Matrix**: Provides a summary of the model’s performance on different classes.
- **Misclassified Images**: Displays images that the model predicted incorrectly, along with the true and predicted labels.

## How to Use

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/handwritten-character-recognition.git
   cd handwritten-character-recognition
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook**:
   - Open the provided Jupyter notebook `handwritten_character_recognition.ipynb` and run all the cells to train the model, evaluate it, and visualize the results.

4. **Use the Saved Model**:
   - The saved model `handwritten_character_recognition_model.h5` can be loaded for inference on new handwritten character data.

## Dataset Sources

- [A-Z Handwritten Alphabets Dataset](https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format)
- [MNIST Dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)

## Results

- The model achieves a high level of accuracy in recognizing both digits and letters.
- The confusion matrix and misclassification plots provide insights into the model's performance.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have suggestions or improvements.
