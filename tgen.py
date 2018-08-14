import argparse
from textgenrnn import textgenrnn


def main():
    # Parse command line
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-n", "--name", type=str,
                            dest="name", default='textgenrnn',
                            help="Name of the network data.")
    args = arg_parser.parse_args()

    textgen = textgenrnn(weights_path=args.name + '_weights.hdf5',
                         vocab_path=args.name + '_vocab.json',
                         config_path=args.name + '_config.json')
    textgen.generate_samples()


if __name__ == "__main__":
    main()
