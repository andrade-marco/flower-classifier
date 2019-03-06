# CLASSIFIER
#
#This file contains functions that help set up the pre-trained network, and
# generate predictions

#Modules import
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from process import process_image

#Building model
def build_model(arch, inputs, hidden_units, output, rate):
    """
    This function builds the model to be used in training or predition using
    the parameters passed
    Params:
        arch - Model architecture/type to be used
        hidden_units - Number of hidden units to be used in classifier
    Returns:
        model - Model with replace classifier and freezed gradients
    """
    fc1_input = inputs if inputs else 25088
    fc2_output = output if output else 102
    dp_rate = rate if rate else 0.2

    #Check if arch is valid and load the correct model
    model_options = ["vgg11", "vgg13", "vgg19"]
    if arch in model_options:
        print("Building model ---> arch: {}".format(arch))
        if arch == "vgg19":
            model = models.vgg19(pretrained=True)
        elif arch == "vgg13":
            model = models.vgg13(pretrained=True)
        else:
            model = models.vgg11(pretrained=True)
    else:
        print("Invalid model: {} --> Using arch: vgg11".format(arch))
        model = models.vgg11(pretrained=True)

    #Freezes gradients
    for param in model.parameters():
        param.requires_grad = False

    #Build classifier and replace it in model
    classifier = nn.Sequential(OrderedDict([
        ("fc1", nn.Linear(fc1_input, hidden_units)),
        ("relu", nn.ReLU()),
        ("drop", nn.Dropout(p=dp_rate)),
        ("fc2", nn.Linear(hidden_units, fc2_output)),
        ("output", nn.LogSoftmax(dim=1))]))

    model.classifier = classifier

    return model


#Training model
def train_model(model, train_loader, valid_loader, learn_rate, epochs, gpu):
    # Define whether to use GPU or CPU and move model
    device = get_device(gpu)
    model.to(device)

    # Configure loss function and optimiter for backprop steps
    # Only the classifier is to be trained
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)

    print("Training in progress...")
    running_loss = 0
    for e in range(epochs):
        for inputs, labels in train_loader:
            # Move inputs and labels to same place model is
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero, or re-initialize, the gradients
            # (important, as gradient accumulate)
            # Feed model forward using the inputs to then calculate
            # the loss (comparing to labels)
            # Given loss, perform backpropagation and gradient descent
            # step using optimizer
            optimizer.zero_grad()
            log_ps = model.forward(inputs)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            # Save current running loss for analysis of training
            running_loss += loss.item()

        else:
            valid_loss = 0
            accuracy = 0

            # Once batch is done, perform the validation using unseen data
            # Turn off gradients for validation - this save memory and
            # computation complexity
            with torch.no_grad():
                model.eval()
                for inputs, labels in valid_loader:
                    # Same as in training, move variables to same place as model
                    inputs, labels = inputs.to(device), labels.to(device)

                    # Perform forward pass for batch and record loss
                    valid_log_ps = model.forward(inputs)
                    batch_loss = criterion(valid_log_ps, labels)
                    valid_loss += batch_loss.item()

                    # Calculate the probilities from the logs, then get
                    # top class; Compare top class to labels to get binaries,
                    # which the mean gives us the accuracy (i.e. how many were
                    #ones out of the total)
                    ps = torch.exp(valid_log_ps)
                    top_ps, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            # Measure validation loss and accuracy
            norm_train_loss = running_loss/len(train_loader)
            norm_valid_loss = valid_loss/len(valid_loader)
            norm_accuracy = accuracy/len(valid_loader)

            print(20*"-")
            print("Epoch - {} of {}...".format(e+1, epochs))
            print("Train loss: {} |  Validation loss: {}".format(norm_train_loss, norm_valid_loss))
            print("Validation accuracy: {}".format(norm_accuracy))

            # Reinitiate loss and revert model back to train mode
            running_loss = 0
            model.train()
    print("Training completed!")
    print(40 * "-", end="\n\n")


def test_model(model, test_loader, gpu):
    # Define whether to use GPU or CPU
    device = get_device(gpu)
    model.to(device)

    print("Testing model...")
    test_accuracy = 0
    with torch.no_grad():
        model.eval()
        for inputs, labels in test_loader:
            # As before, move the variables to same device as model
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass and measuring accuracy
            test_log_ps = model.forward(inputs)
            ps = torch.exp(test_log_ps)
            top_ps, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            norm_test_accuracy = test_accuracy/len(test_loader)

            #Show results
            print("Test accuracy: {}".format(norm_test_accuracy))
    print("Testing completed")
    print(40 * "-", end="\n\n")


# Reload model
def load_checkpoint(filepath, gpu):
    """
    Function to load model checkpoint
    Params:
        filepath - Path to checkpoint file (.pth)
    Returns:
        model - Model loaded according to checkpoint specs
    """
    device = get_device(gpu)
    checkpoint = torch.load(filepath, map_location=device)

    model = build_model(
        checkpoint["arch"],
        checkpoint["fc1_input"],
        checkpoint["fc1_output"],
        checkpoint["fc2_output"],
        checkpoint["dp_rate"])
    model.class_to_idx = checkpoint["class_to_idx"]
    model.load_state_dict(checkpoint["model_state"])

    return model


#Predicting image
def predict(img, model, topk, gpu):
    '''
    Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Move model and image to appropriate device
    device = get_device(gpu)
    model.to(device)
    img.to(device)

    # Reshape image
    img = img.reshape(-1, img.shape[0], img.shape[1], img.shape[2])

    #Turn off gradients and get output from model
    with torch.no_grad():
        output = model.forward(img.float())

    ps = torch.exp(output)
    top_ps, top_class = ps.topk(topk)

    return (top_ps.numpy().squeeze(), top_class.numpy().squeeze())

#Get device
def get_device(gpu):
    device = "cpu"
    if torch.cuda.is_available() and gpu:
        device = "cuda"

    return device
