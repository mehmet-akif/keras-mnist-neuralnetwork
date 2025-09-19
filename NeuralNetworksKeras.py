import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()  

plt.figure(figsize=(5, 2))
for i in range(10):
    plt.subplot(5, 2, i + 1)
    plt.imshow(x_train[i], cmap='gray')  
    plt.axis('off')
plt.show()

print("In the training set, there are", x_train.shape[0], "instances (2D grayscale image data with 28×28 pixels. \
In turn, every image is represented as a 28×28 array rather than a 1D array of size 784. \
Pixel values range from 0 (white) to 255 (black).) \
The associated labels are digits ranging from 0 to 9.")  

x_train = x_train / 255.0  
x_test = x_test / 255.0    

model = keras.models.Sequential()  


model.add(keras.layers.Flatten(input_shape=(28, 28)))  

# First hidden layer (300 neurons, ReLU)
model.add(keras.layers.Dense(300, activation="relu"))  

# Second hidden layer (100 neurons, ReLU)
model.add(keras.layers.Dense(100, activation="relu"))  

# Output layer (10 neurons, softmax)
model.add(keras.layers.Dense(10, activation="softmax"))  

print("The softmax activation function was used for the output layer because it outputs a probability distribution across all classes, \
ensuring the sum of probabilities is 1, which is ideal for multi-class classification.")  

print("The size of the first hidden layer is 300. None means the batch size is flexible. \
The total number of parameters of the first hidden layer is 235,500, which refers to the number of weights plus biases.")  
model.summary()

model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])  

print("An epoch is one full pass through the entire training dataset, during which the model sees every training example once.")  

model.fit(x_train, y_train, epochs=20)  

plt.close('all')
y_pred = model.predict(x_test[:10])  
plt.figure(figsize=(5, 2))
for i in range(10):
    plt.subplot(5, 2, i + 1)
    plt.title('Predicted label: ' + str(np.argmax(y_pred[i])))  
    plt.imshow(x_test[i], cmap='gray') 
    plt.axis('off')
plt.show()
