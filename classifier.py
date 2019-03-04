# CLASSIFIER
#
#This file contains functions that help set up the pre-trained network, and
# generate predictions

def generate_classfier(units=1024):
  classifier = nn.Sequential(OrderedDict([
    ("fc1", nn.Linear(25088, units)),
    ("relu", nn.ReLU()),
    ("drop", nn.Dropout(p=0.2)),
    ("fc2", nn.Linear(units, 102)),
    ("output", nn.LogSoftmax(dim=1))]))

    return classifier
