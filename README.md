# Talky
A character based RNN text generator, inspired by Lucashu1

# Installation
First install [pipenv](https://pypi.python.org/pypi/pipenv), then
`pipenv install`

# Training and Talking
To train the network, run
`pipenv run python train.py <corpus.txt> <number of epochs to train> <load from previous epoch age (optional)>`

This will generate a keras network save file for each epoch under the `saves` directory.  The 'age' of the network is how many epochs it has run.

To have it talk, run
`pipenv run python talk.py <corpus.txt> <age of saved network> <number of characters to generate (default: 100)> <variability (default: .5)>
