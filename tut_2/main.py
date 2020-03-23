import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Load in our dataset, if it's not downloaded yet, it'll download automatically.
data = keras.datasets.fashion_mnist

# Split up our training data and test data
(train_images, train_labels), (test_images, test_labels) = data.load_data()

# Define the names of the classes. The labels are defined as integers, so we link the classes to the labels
# for human interactions.
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Simplify our data to values between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

"""
This will show the image irl
"""
#print(train_images[7])

#plt.imshow(train_images[7], cmap=plt.cm.binary)
#plt.show()

# Create our neural infrastructure. The first layer is our input, the pixels of the image. The between layer is a "hidden layer". We don't control that.
# The third layer is an output layer. This will indicate by how much our network thinks it's that prediction.
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
    ])

# Compiling our network with some algorithm parameters
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Training our model
model.fit(train_images, train_labels, epochs=8)

# Evaluating our model. This will give an indication of how accurate our model's predictions are.
# test_loss, test_acc = model.evaluate(test_images, test_labels)

# Predict our test_images. You can also predict one image by defining it's index.
prediction = model.predict(test_images)

# Loop through 5 images to see the prediction, the actual value and a visual image of the predicted image.
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()

