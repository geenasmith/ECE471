# ECE 471 - Assingment 1
# Geena Smith
# V00835915

import sys, os, cv2

def print_usage():
    print("Usage: \n\tpython3 histmatch.py <input_dir> <output_dir>\n")
    exit()

def color2gray(input_dir,output_dir):
    # check if directory exists -- done
    # open input directory and get the list of images -- done
    # see if output directory exists, if not create it
    # for each image:
    # open image, convert to grayscale
    # save image to the output directory

    # check if input directory exists
    if not(os.path.exists(input_dir)):
        print("Input directory \"%s\" does not exist." %input_dir)
        print_usage()

    # if output directory does not exist, create it
    if not(os.path.exists(output_dir)):
        os.makedirs(output_dir)

    validSuffixes = ['jpg','png']   #need to check format of images
    
    inputFiles = os.listdir(input_dir)
    # print(inputFiles)

    if len(inputFiles) == 0:
        print("No input images to convert.")
        exit()
    
    for fileName in inputFiles:
        print(input_dir+fileName)
        try:
            img = cv2.imread(input_dir+fileName)

            print(img.shape)

            # check if image is rgb
            h,w,c = img.shape

            

            # convert to grayscale
            # save to output directory with same name


        except IOError:
            # do nothing, file is not a valid image
            continue


        # do something
    #     try:
    #         # see if file is an image
    #         img = cv2.imread(input_dir + fileName)
    #         print("successfully read: %s" % fileName)
    #     except:
    #         # not an image

    # loop through input file names, determine valid files to convert, add full directory
    # convert image
    # save to 

def main(argv):
    # need to check input arguments and print usage
    # need to check the format of the input and make sure the directory exists
    if not(len(argv) == 2):
        print("Incorrect number of input arguments.")
        print_usage()
    color2gray(argv[0],argv[1])

if __name__ == "__main__":
    main(sys.argv[1:])