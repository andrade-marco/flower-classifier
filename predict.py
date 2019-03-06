# PREDICT
#
#This file is responsible for making prediction using the trained network
#to classify passed images into flower categories

#Module imports
import json
import matplotlib.pyplot as plt
import numpy as np
from process import parse_inputs, process_image, imshow
from classifier import load_checkpoint, predict

#Main
def main():
    args = parse_inputs(predict=True)

    # Load the model and process image
    # Define whether to use GPU or CPU and move model/img
    model = load_checkpoint(args.checkpoint, args.gpu)
    img = process_image(args.path)

    # Get prediction
    probs, classes = predict(img, model, args.top_k, args.gpu)

    # Load file with category names and classify image
    # Results are printed showing the category names and
    # probabilities
    gap = 40
    precision = 3
    with open(args.category_names, "r") as file:
        print("---- RESULTS ----")
        print("Flower name{}Prob(%)".format((gap - 11) * " "))
        print("-" * (gap + 8))

        flower_dict = json.load(file)
        if classes.ndim < 1:
            name = flower_dict[str(classes)]
            prob = str(round(probs * 100, precision))
            space = "." * (gap - len(name))
            print("{}{}{}%".format(name, space, prob))
        else:
            for idx, val in enumerate(classes.tolist()):
                name = flower_dict[str(val)]
                prob = str(round(probs[idx] * 100, precision))
                space = "." * (gap - len(name))
                print("{}{}{}%".format(name, space, prob))
        print("-" * (gap + 8), end='\n\n')


# Run main function
if __name__ == '__main__':
    main()
