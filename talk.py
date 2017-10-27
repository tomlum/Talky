# Adapted from Lucashu1
import numpy as np
import random
import sys
import codecs
import os.path
from params import *


def main(corpus, age, genLength, diversity):
    from keras.models import load_model

    fileName = (
        "saves/" +
        corpus +
        "-V" + str(version) +
        "-HL" + str(hiddenLayers) +
        "-LR" + str(learningRate) +
        "-BS" + str(batchSize) +
        "-LB" + str(lookBack) +
        "-SK" + str(skip) +
        "-A" + str(age) +
        ".h5"
    )

    ###############################################################
    #  Organize Corpus
    ###############################################################

    text = codecs.open(corpus).read().lower()
    chars = sorted(list(set(text)))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    sentences = []
    next_chars = []
    for i in range(0, len(text) - lookBack, skip):
        sentences.append(text[i: i + lookBack])
        next_chars.append(text[i + lookBack])

    print("Vectorization...")
    X = np.zeros((len(sentences), lookBack, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    print("Building Model...")
    model = load_model(fileName)

    ###############################################################
    #  Generating
    ###############################################################
    print()

    def sample(preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype("float64")
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    print("Diversity is " + str(diversity))

    start_index = random.randint(0, len(text) - lookBack - 1)
    generated = ""
    sentence = text[start_index: start_index + lookBack]

    print("-- Generating with seed: " + sentence)
    sys.stdout.write(generated)
    print("------------------------------------------------")

    for i in range(genLength):
        x = np.zeros((1, lookBack, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()
    print()
    print()


if __name__ == "__main__":
    print()

    args = sys.argv
    validArgs = True

    # Check Corpus
    if (len(args) > 1 and
            (os.path.isfile(args[1]) and (args[1][-3:] == "txt"))):
        corpus = args[1]
    else:
        print("Invalid corpus file")
        validArgs = False

    # Age
    if len(args) > 2:
        age = int(args[2])
        if (not os.path.isfile(
            "saves/" +
            corpus +
            "-V" + str(version) +
            "-HL" + str(hiddenLayers) +
            "-LR" + str(learningRate) +
            "-BS" + str(batchSize) +
            "-LB" + str(lookBack) +
            "-SK" + str(skip) +
            "-A" + str(age) +
            ".h5"
        )):
            print("Save of Age " + str(age) + " does not exist")
            validArgs = False
    else:
        print("Invalid network age")
        validArgs = False

    # Age
    if len(args) > 3:
        genLength = int(args[3])
        if genLength < 1:
            print("Invalid num of characters to generate, must be > 1")
            validArgs = False
    else:
        genLength = 100

    # Diversity
    if len(args) > 4:
        diversity = float(args[4])
        if diversity > 1:
            print("Invalid variability, must be < 1, > 0")
            validArgs = False
    else:
        diversity = .5

    if(validArgs):
        main(corpus, age, genLength, diversity)
    else:
        print()
        print("Command should look like:")
        print(
            "talk.py <corpus.txt> <age of network> <characters to generate> <variability>")
        print()
