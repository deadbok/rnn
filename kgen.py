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
file_list = ["romeo_and_juliet.txt"]
vocab_file = "words_vocab.pkl"
sequences_step = 1  # step to create sequences

rnn_size = 256  # size of RNN
seq_length = 30  # sequence length
learning_rate = 0.001  # learning rate

batch_size = 32  # minibatch size
num_epochs = 100  # number of epochs

#load vocabulary
print("loading vocabulary...")
vocab_file = "words_vocab.pkl"

with open('words_vocab.pkl', 'rb') as f:
        words, vocab, vocabulary_inv = cPickle.load(f)

vocab_size = len(words)

from keras.models import load_model
# load the model
print("loading model...")
model = load_model('my_model_generate_sentences.h5')

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

words_number = 50 # number of words to generate
seed_sentences = "i watch her move " #seed sentence to start the generating.

#initiate sentences
generated = ''
sentence = []

#we shate the seed accordingly to the neural netwrok needs:
for i in range (seq_length):
    sentence.append("a")

seed = seed_sentences.split()

for i in range(len(seed)):
    sentence[seq_length-i-1]=seed[len(seed)-i-1]

generated += ' '.join(sentence)

#the, we generate the text
for i in range(words_number):
    #create the vector
    x = np.zeros((1, seq_length, vocab_size))
    for t, word in enumerate(sentence):
        x[0, t, vocab[word]] = 1.

    #calculate next word
    preds = model.predict(x, verbose=0)[0]
    next_index = sample(preds, 0.33)
    next_word = vocabulary_inv[next_index]

    #add the next word to the text
    generated += " " + next_word
    # shift the sentence by one, and and the next word at its end
    sentence = sentence[1:] + [next_word]

#print the whole text
print(generated)

