# MNIST Handwritten Digit Recognition with Keras

This project implements a **neural network using Keras and TensorFlow** to recognize handwritten digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).  
It was developed as part of a **Machine Learning assignment** to gain hands-on experience with building, training, and evaluating neural networks.

---

## 🚀 Features
- Loads and visualizes the MNIST dataset (60,000 training, 10,000 test images).
- Preprocesses data by normalizing pixel values (0–255 → 0–1).
- Implements a **Sequential neural network**:
  - Flatten layer (28×28 → 784)
  - Dense hidden layer (300 neurons, ReLU)
  - Dense hidden layer (100 neurons, ReLU)
  - Dense output layer (10 neurons, Softmax)
- Trains the model for 20 epochs using **Stochastic Gradient Descent (SGD)**.
- Evaluates model accuracy on the test set.
- Visualizes the first 10 test predictions.


