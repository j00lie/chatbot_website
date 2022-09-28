import json
from nltk_utils import tokenize, stem, bag_off_words
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout
from keras.optimizers import SGD
import random
import pickle

with open("intents.json", "r") as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)
    for pattern in intent["patterns"]:
        w = tokenize(pattern)
        all_words.extend(w)  # tokenize returns an array so use extend
        xy.append((w, tag))  # get pattern and corresponding tag

ignore_words = ["?", "!", ".", ","]
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))  # sort and remove duplicates
tags = sorted(set(tags))

# saving the words and classes list to binary files
pickle.dump(all_words, open("all_words.pkl", "wb"))
pickle.dump(tags, open("tags.pkl", "wb"))


X_train = []
Y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_off_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    Y_train.append(label)

both_lists = list(zip(X_train, Y_train))  # combine lists
random.shuffle(both_lists)
X_train, Y_train = zip(*both_lists)  # part lists

X_train = np.array(X_train)
Y_train = np.array(Y_train)

Y_train = tf.keras.utils.to_categorical(Y_train, 7)


model = Sequential(
    [
        Dense(128, input_shape=(len(all_words),), activation="relu"),
        Dropout(0.1),
        Dense(64, activation="relu"),
        Dropout(0.1),
        Dense(len(tags), activation="softmax"),
    ]
)

# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(x=X_train, y=Y_train, epochs=200, batch_size=5)


model.save("chatbot")
