# ECE 471 - Assingment 1
# Geena Smith
# V00835915

import sys, os

def print_usage():
    print("Usage: \n\tpython3 histmatch.py <input_dir> <output_dir>\n")
    exit()

def color2gray(input_dir,output_dir):
    # check if directory exists -- done
    # open input directory and get the list of images
    # see if output directory exists, if not create it
    # for each image:
    # open image, convert to grayscale
    # save image to the output directory

    if not(os.path.exists(input_dir)):
        print("input directory \"%s\" does not exist." %input_dir)
        print_usage()
    print('... do something ...')

def main(argv):
    # need to check input arguments and print usage
    # need to check the format of the input and make sure the directory exists
    if not(len(argv) == 2):
        print("Incorrect number of input arguments.")
        print_usage()
    color2gray(argv[0],argv[1])

if __name__ == "__main__":
    main(sys.argv[1:])