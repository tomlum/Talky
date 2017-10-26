# Adapted from Lucashu1
from keras.models import load_model
import numpy as np
import random
import sys
import codecs

corpus = "speare-short.txt"
fileName = "speare-short.txt-V1-HL128-LR0.001-BS128-LB20-SK3-A1.h5"
genLength = 500
diversity = .5

lookBack = 20
skip = 3

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

print("----- Generating with seed: " + sentence)
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
