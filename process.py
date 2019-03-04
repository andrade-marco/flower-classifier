# PROCESS
#
#This file contains utility functions for processing data for running the
#train and predict files

#Module imports
import argparse
import numpy as np
import torch
from PIL import Image

#Parse input arguments
def parse_inputs(predict=False):
    """
    Function to retrieve the command line arguments passed by the user. These
    arguments are used to set up the training of the model/neural network.
    Command line args:
        data_dir - Path to files to train and validate network
        save_dir - Path to save model checkpoints
        arch - Model architecture to be used (vgg11, vgg13 or vgg19)
        learning_rate - Learning rate to be used by model
        hidden_units - Number of hidden units to be used in model
        epochs - Number of epochs to train the model
        gpu - Boolean indicating whether or not to use GPU for training
    Params:
        None
    Returns:
        parse_args() - Command line args object
    """
    parser = argparse.ArgumentParser()

    if predict:
        parser.add_argument("path")
        parser.add_argument("checkpoint",
                            default="checkpoint_vgg11.pth")
        parser.add_argument("--top_k", type=int, default=5)
        parser.add_argument("--category_names",
                            type=str,
                            default="cat_to_name.json")
    else:
        parser.add_argument("data_dir", default="flowers")
        parser.add_argument("--save_dir", type=str, default="")
        parser.add_argument("--arch", type=str, default="vgg11")
        parser.add_argument("--learning_rate", type=float, default=0.02)
        parser.add_argument("--hidden_units", type=int, default=1024)
        parser.add_argument("--epochs", type=int, default=30)

    parser.add_argument("--gpu", action="store_true")
    return parser.parse_args()


#Process image
def process_image(image):
    """
    Function that scales, crops, and normalizes a PIL image for a PyTorch model
    Params:
        image - String containing the path to an image
    Returns:
        torch.from_numpy() - Tensor to be used for predictions
    """
    # Get image size and aspect ratio
    img = Image.open(image)
    width, height = img.size
    ratio = width/height

    # Calculate new dimensions for resizing and cropping
    w = 256 if ratio < 1 else 256*ratio
    h = 256/ratio if ratio < 1 else 256
    l = (w - 224)/2
    r = (w + 224)/2
    u = (h - 224)/2
    b = (h + 224)/2

    # Resize image, then crop out the center (224 x 224)
    # Get the np array of the image to normalize it
    img.thumbnail((w, h))
    cropped_img = img.crop((l, u, r, b)) #left, upper, right, lower
    np_img = np.array(cropped_img)
    norm_np_img = ((np_img/255) - [0.485, 0.456, 0.406])/[0.229, 0.224, 0.225]

    # Return the tensor
    return torch.from_numpy(norm_np_img.transpose())

#Show image
def imshow(image, ax=None, title=None):
    """
    Imshow for Tensor.
    Params:
        image - Tensor containing image data
        ax - Axis object
        title - Image title
    """
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise
    # when displayed
    image = np.clip(image, 0, 1)
    ax.imshow(image)

    return ax