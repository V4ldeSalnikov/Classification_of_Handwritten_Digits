# MNIST Classification using Machine Learning Models

## Project Overview

This project demonstrates the application of **machine learning algorithms** for classifying the **MNIST dataset** using a combination of models from **scikit-learn** and **TensorFlow**. The MNIST dataset contains images of handwritten digits (0-9), and the goal is to classify these images accurately.

The project includes:

- **Data preprocessing** using normalization and reshaping.
- Implementation of multiple machine learning models, including:
  - K-Nearest Neighbors
  - Decision Tree Classifier
  - Logistic Regression
  - Random Forest Classifier
- **Hyperparameter tuning** using **GridSearchCV**.
- Model evaluation based on **accuracy score**.

## Dataset

The **MNIST dataset** is loaded directly using TensorFlow's `keras.datasets` module. The dataset contains 70,000 28x28 pixel grayscale images of handwritten digits. The dataset is split into training and testing sets.

- Training data: 60,000 images
- Test data: 10,000 images

For the purposes of this project, a smaller subset of the data is used with **6,000 images** for training and evaluation.

## Code Overview

1. **Data Loading**: 
   - The MNIST dataset is loaded using TensorFlow's built-in `keras.datasets.mnist` loader.

2. **Data Preprocessing**:
   - The images are reshaped into 1D arrays of size 28x28 = 784 for each image.
   - The features are **normalized** to improve model performance.

3. **Model Training and Evaluation**: 
   The following models are trained on the normalized data:
   - **K-Nearest Neighbors** (KNN)
   - **Decision Tree Classifier**
   - **Logistic Regression**
   - **Random Forest Classifier**

4. **Hyperparameter Tuning**: 
   **GridSearchCV** is used to find the best hyperparameters for the K-Nearest Neighbors and Random Forest models, tuning parameters such as:
   - **K-Nearest Neighbors**: `n_neighbors`, `weights`, and `algorithm`
   - **Random Forest**: `n_estimators`, `max_features`, and `class_weight`

5. **Model Evaluation**: 
   The models are evaluated using **accuracy** metrics, and the best models are displayed along with their tuned parameters.

