import argparse
import os
from textgenrnn import textgenrnn


model_cfg = {
    'rnn_size': 128,
    'rnn_layers': 6,
    'rnn_bidirectional': True,
    'max_length': 5,
    'max_words': 10000,
    'dim_embeddings': 100,
    'word_level': True,
}

train_cfg = {
    'line_delimited': True,
    'num_epochs': 50,
    'gen_epochs': 5,
    'batch_size': 256,
    'train_size': 0.8,
    'dropout': 0.0,
    'max_gen_length': 30,
    'validation': False,
    'is_csv': False,
}


def nonblank_lines(f):
    for l in f:
        line = l.rstrip()
        if line:
            yield line


def main():
    # Parse command line
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-n", "--name", type=str,
                            dest="name", default='textgenrnn',
                            help="Name of the output network data.")
    arg_parser.add_argument("corpusfiles", type=str, nargs='+', default=['corpus.txt'],
                            help="Text files to process.")
    arg_parser.add_argument('-e', '--epochs', type=int, dest='epochs', default=50,
                            help="Epochs to train the network.")
    args = arg_parser.parse_args()

    textgen = textgenrnn(name=args.name + '-text')

    for corpus_file_name in args.corpusfiles:
        lines = []
        with open(corpus_file_name) as corpus_file:
            for line in nonblank_lines(corpus_file):
                lines.append(line.lstrip(' '))
        textgen.train_on_texts(
            texts=lines,
            new_model=True,
            num_epochs=args.epochs,
            gen_epochs=train_cfg['gen_epochs'],
            batch_size=train_cfg['batch_size'],
            train_size=train_cfg['train_size'],
            dropout=train_cfg['dropout'],
            max_gen_length=train_cfg['max_gen_length'],
            validation=train_cfg['validation'],
            is_csv=train_cfg['is_csv'],
            rnn_layers=model_cfg['rnn_layers'],
            rnn_size=model_cfg['rnn_size'],
            rnn_bidirectional=model_cfg['rnn_bidirectional'],
            max_length=model_cfg['max_length'],
            dim_embeddings=model_cfg['dim_embeddings'],
            word_level=model_cfg['word_level'])

    textgen.generate_samples()
    print(textgen.model.summary())


if __name__ == "__main__":
    main()
