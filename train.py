# TRAIN
#
#This file is responsible for training the network that classifies
#images into the different flower categories

#Module imports
from process import parse_inputs, get_data_loaders
from classifier import build_model

#Main
def main():
    args = parse_inputs()
    train, valid, test = get_data_loaders(args["data_dir"])
    model = build_model(
        args["arch"], None, args["hidden_units"], None, args["learn_rate"])


# Run main function
if __name__ == '__main__':
    main()
