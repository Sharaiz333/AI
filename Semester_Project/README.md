Handwritten Digit Recognition System

This project implements a handwritten digit recognition system using the MNIST dataset. It combines both traditional machine learning and deep learning approaches, along with a graphical user interface for real-time digit prediction.

Project Overview

The system is designed to recognize digits (0–9) from user input or dataset images. Two models are implemented:

K-Nearest Neighbors (KNN)
Convolutional Neural Network (CNN)

A GUI built with Tkinter allows users to draw digits and get predictions from either model.

Dataset
Dataset: MNIST (CSV format from Kaggle)
Total samples: 70,000
Training: 60,000
Testing: 10,000
Image size: 28 × 28 pixels
Format: Grayscale (0–255)

Each image is flattened into 784 features in the CSV file.

Features
Data loading and preprocessing pipeline
Normalization of pixel values (0–255 → 0–1)
KNN model with hyperparameter tuning (best k selection)
CNN model with:
Convolutional layers
Batch normalization
Dropout for regularization
Data augmentation using ImageDataGenerator
Model saving and loading
Performance evaluation:
Accuracy
Confusion matrix
GUI for real-time digit drawing and prediction
Project Structure
project/
│
├── data/
│   ├── mnist_train.csv
│   └── mnist_test.csv
│
├── dataset.py
├── preprocessing.py
├── knn_model.py
├── nn_model.py
├── train.py
├── evaluate.py
├── gui.py
│
├── knn_model.pkl
├── nn_model.h5
│
├── confusion_matrix.png
├── confusion_matrix_nn.png
├── sample_predictions.png
│
└── README.md
Installation

Install required libraries:

pip install numpy pandas matplotlib scikit-learn tensorflow pillow
How to Run
1. Train Models
python train.py

This will:

Load dataset
Preprocess data
Train KNN and CNN models
Save models as .pkl and .h5
2. Evaluate Models
python evaluate.py

This will:

Calculate accuracy
Generate confusion matrices
Save prediction visualizations
3. Run GUI
python gui.py
Draw a digit (0–9)
Select model (KNN or Neural Network)
Click Predict
Model Details
K-Nearest Neighbors (KNN)
Uses distance-weighted voting
Optimal k selected using cross-validation
Works on flattened pixel vectors
Convolutional Neural Network (CNN)
Input: 28×28×1 images
Layers:
Conv2D + ReLU
Batch Normalization
MaxPooling
Dropout
Dense layers
Output: Softmax (10 classes)
Evaluation
KNN evaluated on subset of test data
CNN evaluated on full dataset
Metrics:
Accuracy
Confusion Matrix
GUI Features
Interactive drawing canvas
Real-time preprocessing
Model selection (KNN or CNN)
Confidence score display
Input validation for digit quality
Limitations
MNIST is a simple dataset (not real-world complexity)
KNN is slower for large datasets
CNN requires more computational resources
GUI input may vary in quality
Authors
Sharaiz Ahmed
Umair Waseem

BSCS – Riphah International University

Future Improvements
Add support for custom image input
Improve GUI drawing accuracy
Deploy as a web application
Use more advanced architectures (e.g., ResNet)
