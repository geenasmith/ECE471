# ECE 471 - Assingment 1
# Geena Smith
# V00835915

import sys, os, cv2, numpy

def print_usage():
    print("Usage: \n\tpython3 histmatch.py <input_dir> <output_dir>\n")
    exit()

def color2gray(input_dir,output_dir):
    # check if directory exists -- done
    # open input directory and get the list of images -- done
    # see if output directory exists, if not create it -- done
    # for each image:
    # open image, convert to grayscale
    # save image to the output directory

    # add '/' to end of directory if it is not included
    if not(input_dir[-1] == '/'):
        input_dir = input_dir + '/'

    # check if input directory exists
    if not(os.path.exists(input_dir)):
        print("Input directory \"%s\" does not exist." %input_dir)
        print_usage()

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
    
    # loop through files in the directory
    for fileName in inputFiles:
        print("Image: " + input_dir + fileName)
        # check if the file is an image by trying to read it
        try:
            # docs: https://docs.opencv.org/4.2.0/d4/da8/group__imgcodecs.html
            # RGB channels is in order B,G,R
            img = cv2.imread(input_dir+fileName)

            # check if image is rgb
            h,w,c = img.shape
            print("Image dimensions: %d,%d,%d" % (h,w,c)) #DELETE ME
            
            if not(c == 3):
                # could not read image or image is not RGB
                continue

            # normalize RGB range to [0,255]
            cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)

            # DELETE ME
            ctrimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            cv2.imwrite(output_dir+"ctrl"+fileName,ctrimg)

            # convert to gray
            # use RGB weights of [0.07,0.72,0.21]
            grayimg = 0.21*img[:,:,0]+0.72*img[:,:,1]+0.07*img[:,:,2]

            cv2.imwrite(output_dir+fileName,grayimg)


        except IOError:
            # do nothing, file is not a valid image
            continue


def main(argv):
    # need to check input arguments and print usage
    # need to check the format of the input and make sure the directory exists
    if not(len(argv) == 2):
        print("Incorrect number of input arguments.")
        print_usage()
    color2gray(argv[0],argv[1])

if __name__ == "__main__":
    main(sys.argv[1:])