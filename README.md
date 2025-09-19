# MNIST Handwritten Digit Recognition with Keras

## Overview

This project demonstrates handwritten digit recognition using a **neural network built with Keras and TensorFlow**.  
The model is trained on the **MNIST dataset** of 70,000 grayscale images of digits (0–9).  
The project was completed as part of my **Machine Learning course assignment** and showcases a practical introduction to deep learning for image classification.

---

## Table of Contents
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Preprocessing Steps](#preprocessing-steps)
- [Methodology](#methodology)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Usage](#usage)
- [Files Included](#files-included)

---

## Project Description

The goal of this project is to classify handwritten digits (0–9) from the MNIST dataset using a **feedforward neural network**.  
This task demonstrates the power of deep learning in computer vision and serves as a foundation for more advanced models such as **Convolutional Neural Networks (CNNs)**.

---

## Dataset

- **Source**: MNIST dataset (built into Keras).
- **Details**:
  - Training set: 60,000 images
  - Test set: 10,000 images
  - Image size: 28×28 pixels, grayscale
  - Labels: digits `0–9`

---

## Preprocessing Steps

1. Loaded the dataset from `keras.datasets.mnist`.
2. Visualized the first 10 digits for verification.
3. Normalized pixel values from range `0–255` to range `0–1`.
4. Converted 2D image data (28×28) into 1D arrays for neural network input using a **Flatten layer**.

---

## Methodology

1. **Model Architecture**:
   - Flatten layer (28×28 → 784)
   - Dense hidden layer (300 neurons, ReLU activation)
   - Dense hidden layer (100 neurons, ReLU activation)
   - Dense output layer (10 neurons, Softmax activation)

2. **Training Setup**:
   - Loss function: `sparse_categorical_crossentropy`
   - Optimizer: Stochastic Gradient Descent (SGD)
   - Epochs: 20

3. **Evaluation**:
   - Accuracy measured on the test dataset.
   - Predictions generated for the first 10 test images.

---

## Results

- The network achieved **~97–98% accuracy** on the MNIST test dataset.  
- Predicted labels for sample digits aligned well with actual labels.  
- Visualization confirmed correct classification of most test digits.

---

## Technologies Used

- **Languages**: Python
- **Libraries**:
  - TensorFlow / Keras
  - NumPy
  - Matplotlib

---

## Usage

1. Clone the repository:
   ```bash
      git clone https://github.com/<mehmet-akif>/mnist-keras-classifier.git

   git clone https://github.com/<mehmet-akif>/mnist-keras-classifier.git
   cd mnist-keras-classifier
   
