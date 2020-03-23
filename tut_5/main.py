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

