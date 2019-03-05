# TRAIN
#
#This file is responsible for training the network that classifies
#images into the different flower categories

#Module imports
import torch
from process import parse_inputs, get_data_loaders
from classifier import build_model, train_model, test_model

#Main
def main():
    args = parse_inputs()

    # Get data loaders for training and build the model
    loaders = get_data_loaders(args.data_dir)
    train_set = loaders["train_set"]
    train, valid, test = loaders["loaders"]
    model = build_model(args.arch, None, args.hidden_units, None, None)

    # Train and test model
    train_model(model, train, valid, args.learn_rate, args.epochs, args.gpu)
    test_model(model, test, args.gpu)

    # Saving checkpoint
    model_options = ["vgg11", "vgg13", "vgg19"]
    checkpoint = {"fc1_input": 25088,
                  "fc1_output": args.hidden_units,
                  "fc2_output": 102,
                  "dp_rate": 0.2,
                  "epochs": args.epochs,
                  "model_state": model.state_dict(),
                  "class_to_idx": train_set.class_to_idx}

    if args.arch in model_options:
        checkpoint["arch"] = args.arch
        filename = "checkpoint_{}.pth".format(args.arch)
    else:
        checkpoint["arch"] = "vgg11"
        filename = "checkpoint_vgg11.pth"

    torch.save(checkpoint, filename)
    print("Model checkpoint saved!")
    print("You can now use the predict.py script to classify flower images")

# Run main function
if __name__ == '__main__':
    main()
