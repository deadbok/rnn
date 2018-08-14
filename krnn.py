#!/bin/env python3

# import Keras library
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, Input, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.metrics import categorical_accuracy

# import spacy, and spacy french model
# spacy is used to work on text
import spacy
nlp = spacy.load('en')

# import other libraries
import numpy as np
import random
import sys
import os
import time
import codecs
import collections
from six.moves import cPickle

# define parameters used in the tutorial
file_list = ["db_lyrics.txt"]
vocab_file = "words_vocab.pkl"
sequences_step = 1  # step to create sequences

rnn_size = 256  # size of RNN
seq_length = 30  # sequence length
learning_rate = 0.001  # learning rate

batch_size = 32  # minibatch size
num_epochs = 100  # number of epochs


def create_wordlist(doc):
    wl = []
    for word in doc:
        text = word.text.strip('\n')
        if text not in ('', '\u2009', '\xa0'):
            wl.append(text.lower())
    return wl


wordlist = []

print("Reading text corpus")
for file_name in file_list:
    # read data
    with codecs.open(file_name, "r") as f:
        data = f.read()

    # create sentences
    doc = nlp(data)
    wl = create_wordlist(doc)
    wordlist = wordlist + wl


print("Creating vocabulary")
# count the number of words
word_counts = collections.Counter(wordlist)

# Mapping from index to word : that's the vocabulary
vocabulary_inv = [x[0] for x in word_counts.most_common()]
vocabulary_inv = list(sorted(vocabulary_inv))

# Mapping from word to index
vocab = {x: i for i, x in enumerate(vocabulary_inv)}
words = [x[0] for x in word_counts.most_common()]

# size of the vocabulary
vocab_size = len(words)
print("vocab size: ", vocab_size)

# save the words and vocabulary
with open(os.path.join(vocab_file), 'wb') as f:
    cPickle.dump((words, vocab, vocabulary_inv), f)

print("Creating sequences")
# create sequences
sequences = []
next_words = []
for i in range(0, len(wordlist) - seq_length, sequences_step):
    sequences.append(wordlist[i: i + seq_length])
    next_words.append(wordlist[i + seq_length])

print('nb sequences:', len(sequences))

X = np.zeros((len(sequences), seq_length, vocab_size), dtype=np.bool)
y = np.zeros((len(sequences), vocab_size), dtype=np.bool)
for i, sentence in enumerate(sequences):
    for t, word in enumerate(sentence):
        X[i, t, vocab[word]] = 1
    y[i, vocab[next_words[i]]] = 1


def bidirectional_lstm_model(seq_length, vocab_size):
    print('Build LSTM model.')
    model = Sequential()
    model.add(Bidirectional(LSTM(rnn_size, activation="relu"),
                            input_shape=(seq_length, vocab_size)))
    model.add(Dropout(0.6))
    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))

    optimizer = Adam(lr=learning_rate)
    callbacks = [EarlyStopping(patience=2, monitor='val_loss')]
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, metrics=[categorical_accuracy])
    print("model built!")
    return model


md = bidirectional_lstm_model(seq_length, vocab_size)
md.summary()

callbacks = [EarlyStopping(patience=4, monitor='val_loss'),
             ModelCheckpoint('my_model_gen_sentences.{epoch:02d}-{val_loss:.2f}.hdf5',
                             monitor='val_loss', verbose=1, mode='auto', period=2)]
# fit the model
history = md.fit(X, y,
                 batch_size=batch_size,
                 shuffle=True,
                 epochs=num_epochs,
                 callbacks=callbacks,
                 validation_split=0.1)

# save the model
md.save('my_model_generate_sentences.h5')
