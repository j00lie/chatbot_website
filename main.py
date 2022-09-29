"""
Sources: 

https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077

https://www.youtube.com/watch?v=k1SzvvFtl4w

https://www.geeksforgeeks.org/deploy-a-chatbot-using-tensorflow-in-python/

"""


from ast import While
import random
import json
from nltk_utils import bag_off_words, tokenize
from keras.models import load_model
import pickle
import numpy as np


with open("intents.json", "r") as f:
    intents = json.load(f)

all_words = pickle.load(open("all_words.pkl", "rb"))
tags = pickle.load(open("tags.pkl", "rb"))
model = load_model("chatbot")


bot_name = "Keijo"


def get_response(message):

    sentence = tokenize(message)
    bow = bag_off_words(sentence, all_words)
    bow = bow.reshape(1, bow.shape[0])

    output = model.predict(bow)  # get probabilities for all tags
    # print(output)
    prediction = np.argmax(output)  # choose highest probability
    # print(prediction)
    tag = tags[prediction]  # get tag corresponging to the probability
    # print(tag)

    if np.max(output) > 0.5:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return random.choice(intent["responses"])
    else:
        return "I dont understand.."


def main():
    print("Lets chat! Type 'quit' to exit")

    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break
        answer = get_response(sentence)
        print(answer)
