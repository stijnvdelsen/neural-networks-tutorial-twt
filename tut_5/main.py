"""
This program looks at a IMDB review and checks if the review is positive or negative.
This program was part of the neural network tutorial of Tech With Tim
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)

# As you can see this data exists out of integer encoded words. Each integer stands for a word
# print(test_data)

# This prints a key value array of word index
word_index = data.get_word_index()

# This will add +3 to every value, because we want to set some presets, see below.
word_index = {k:(v+3) for k, v in word_index.items()}

# Give our presets an index
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

# This will swap our values and keys. Right now the word_index is as follows: "Word": 652. We want it to be: 652: "Word"
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Because all of the reviews are differenct lengths, we need te preprocess our data. We need a fixed amount of input neurons, we choose for this purpose 250.
# We can do this with the function below. We want to cut off the words that are over our max length and we want to add placeholders to reviews that are to short.
# The first paramater is the data we want to preprocess.
# The second parameter is our placeholder for when the function is to short.
# The fourth parameter is the place we want to add our placeholders.
# The fifth parameter is the max length of our data
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

def decode_review(text):
    """
    decode_review decodes the given text to human readable text.

    Outputs a string of human readable text
    """
    return " ".join([reverse_word_index.get(i, "?") for i in text])

# Here we can decode the review into human readable text
# print(decode_review(test_data[1]))

# We want our model down here.

# Create our model
model = keras.Sequential()

# Add the neural layers
model.add(keras.layers.Embedding(10000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

# I need to look up what this does.
model.summary()

# This will compile our model. The loss binary_crossentropy means that we have two possible outcomes.
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# We want to split up our training data for validating.
x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

# This will train our model. The param batch_size means how many reviews we are going to load into memory at once. Verbose means how detailed the ouput is to be printed.
fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

# This will test our model on loss and accuracy
results = model.evaluate(test_data, test_labels)

print(results)

# See the results below:

test_review = test_data[0]
predict = model.predict([test_review])
print("Review: ")
print(decode_review(test_review))

# We want to do a argmax on the prediction, otherwise we'll get a result that we can't compare as easy to the actual label
print("Prediction: " + str(np.argmax(predict[0])))
print("Actual: " + str(test_labels[0]))
