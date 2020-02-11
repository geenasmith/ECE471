# ECE 471 - Assingment 1
# Geena Smith
# V00835915

import sys, os, cv2, numpy

def print_usage():
    print("Usage: \n\tpython3 tograyscale.py <input_dir> <output_dir>\n")
    exit()

def color2gray(input_dir,output_dir):

    # add '/' to end of input directory if it is not included
    if not(input_dir[-1] == '/'):
        input_dir = input_dir + '/'

    # check if input directory exists
    if not(os.path.exists(input_dir)):
        print("Input directory \"%s\" does not exist." %input_dir)
        print_usage()

    # add '/' to end of output directory if it is not included
    if not(output_dir[-1] == '/'):
        output_dir = output_dir + '/'

    # if output directory does not exist, create it
    if not(os.path.exists(output_dir)):
        os.makedirs(output_dir)

    # get list of files present in the directory
    inputFiles = os.listdir(input_dir)

    # no images in the directory
    if len(inputFiles) == 0:
        print("No input images to convert.")
        exit()

    convertedCount = 0  # number of images converted from RGB to grayscale
    
    # loop through files in the directory
    for fileName in inputFiles:
        # check if the file is an image by trying to read it
        try:
            # docs: https://docs.opencv.org/4.2.0/d4/da8/group__imgcodecs.html
            # RGB channels is in order B,G,R
            img = cv2.imread(input_dir+fileName)

            # check if image is rgb
            h,w,c = img.shape

            if not(c == 3):
                # could not read image or image is not RGB
                continue

            # normalize RGB range to [0,255]
            cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)

            # convert to gray
            # use RGB weights of [0.114,0.587,0.299] taken from OpenCV documentation:
            # https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html?highlight=cvtcolor
            grayimg = 0.114*img[:,:,0]+0.587*img[:,:,1]+0.299*img[:,:,2]

            cv2.imwrite(output_dir+fileName,grayimg)
            print("Converted: " + input_dir + fileName)

            convertedCount = convertedCount + 1

        except:
            # do nothing, file is not a valid image
            continue

    print("Done. Converted %d images to grayscale." % convertedCount)


def main(argv):
    if not(len(argv) == 2):
        print("Incorrect number of input arguments.")
        print_usage()
    color2gray(argv[0],argv[1])

if __name__ == "__main__":
    main(sys.argv[1:])