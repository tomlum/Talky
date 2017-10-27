# Adapted from Lucashu1
import sys
import codecs
import os.path
from params import *


def main(corpus, epochs, age):

    from keras.models import Sequential
    from keras.layers import Dense, Activation
    from keras.layers import LSTM
    from keras.optimizers import RMSprop
    from keras.models import load_model
    import numpy as np

    ###############################################################
    #  Organize Corpus
    ###############################################################
    print("Corpus:", corpus)

    text = codecs.open(corpus).read().lower()
    print("Corpus Length:", len(text))

    chars = sorted(list(set(text)))
    print("Total Chars:", len(chars))

    char_indices = dict((c, i) for i, c in enumerate(chars))

    sentences = []
    next_chars = []
    for i in range(0, len(text) - lookBack, skip):
        sentences.append(text[i: i + lookBack])
        next_chars.append(text[i + lookBack])
    print("Total Sequences:", len(sentences))

    print("Vectorization...")
    X = np.zeros((len(sentences), lookBack, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    ###############################################################
    #  Build LSTM
    ###############################################################

    if age > 0:
        print("Loading Model, Age:" + str(age) + "...")
        # Load previous save
        model = load_model(
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
    else:
        print("Building Model...")
        # Or construct new LSTM
        model = Sequential()
        model.add(LSTM(hiddenLayers, input_shape=(lookBack, len(chars))))
        model.add(Dense(len(chars)))
        model.add(Activation("softmax"))
        optimizer = RMSprop(lr=learningRate)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer)

    ###############################################################
    #  Training
    ###############################################################
    print()
    for iteration in range(1, epochs + 1):
        age = age + 1
        print("Iteration ", iteration)
        model.fit(X, y,
                  batch_size=batchSize,
                  epochs=1,
                  verbose=2)
        model.save(
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

    # Number of Epochs
    if len(args) > 2:
        epochs = int(args[2])
    else:
        print("Invalid number of epochs")
        validArgs = False

    # Age
    if len(args) > 3:
        age = int(args[3])
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
        age = 0

    if(validArgs):
        main(corpus, epochs, age)
    else:
        print()
        print("Command should look like:")
        print("'train.py <corpus.txt> <epochs> <starting age of network>'")
        print()
