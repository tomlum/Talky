# Adapted from Lucashu1
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.models import load_model
import numpy as np
import codecs

corpus = "speare-short.txt"
version = 1
# How many epochs its run through
age = 0
hiddenLayers = 128
learningRate = 0.001
batchSize = 128
# How many characters back to consider when generating the next
lookBack = 20
# How many characters to skip after a read in the document
skip = 3

epochs = 50

###############################################################
#  Organize Corpus
###############################################################

text = codecs.open(corpus).read().lower()
print("Corpus Length:", len(text))

chars = sorted(list(set(text)))
print("Total Chars:", len(chars))

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

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
        corpus +
        "-V" + str(version) +
        "-HL" + str(hiddenLayers) +
        "-LR" + str(learningRate) +
        "-BS" + str(batchSize) +
        "-LB" + str(lookBack) +
        "-SK" + str(skip) +
        "-A" + str(age) +
        ".h5")
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

for iteration in range(1, epochs):
    age = age + 1
    print("Iteration ", iteration)
    model.fit(X, y,
              batch_size=batchSize,
              epochs=1,
              verbose=2)
    model.save(
        corpus +
        "-V" + str(version) +
        "-HL" + str(hiddenLayers) +
        "-LR" + str(learningRate) +
        "-BS" + str(batchSize) +
        "-LB" + str(lookBack) +
        "-SK" + str(skip) +
        "-A" + str(age) +
        ".h5")
print()
