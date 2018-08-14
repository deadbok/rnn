# RNN experiments

## Steps

* Get words in corpus and generate a dictionary of all words mapped to integers
* Add pronunciations using pronouncingpy
* Teach a RNN using the corpus mapped to numbers by the dictionary of words
* Use pronouncingpy to add references to rhyming words in the dictionary
* Teach a RNN using the pronunciation of rhyming words

## Pipelines

* RNN taught using the words of the corpus
* RNN taught using the pronunciation of rhyming words

## Words

Words are stored in a dictionary and referenced by an integer during the learning
and generation phases. The dictionary stores the following data for every corpus
word:

* The word itself
* Pronunciations
* Rhymes
