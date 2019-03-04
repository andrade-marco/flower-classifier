# PREDICT
#
#This file is responsible for making prediction using the trained network
#to classify passed images into flower categories

#Module imports
from process import parse_inputs

#Main
def main():
    args = parse_inputs(predict=True)
    print(args)

# Run main function
if __name__ == '__main__':
    main()
