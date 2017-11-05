# Talky
A character based RNN text generator, inspired by Lucashu1

# Installation
First install [pipenv](https://pypi.python.org/pypi/pipenv), then run

`pipenv install`

# Training and Talking
To train the network, run

`pipenv run python train.py <corpus.txt> <number of epochs to train>`

This will generate a keras network save file in the `saves` directory of the weights that produce the least error.  If a save exists already, the network will load the save and continue training.

To have it talk, run

`pipenv run python talk.py <corpus.txt> <number of characters to generate (default: 100)> <variability (default: .5)>`

# Tweaking Parameters
To tweak the parameters of the network, check out params.py.

Note: any changes to parameters (except for `patience`) will cause a separate save filed to be generated/loaded.
