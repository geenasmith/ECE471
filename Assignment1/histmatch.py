# ECE 471 - Assingment 1
# Geena Smith
# V00835915

# using the returned mapping function F, transform input_image

import sys, os, cv2, matplotlib.pyplot as plt, numpy as np
from match_histograms import match_histograms

# 1. create linear cumulative histogram Href
# 2. use match_histograms function to create a mapping function F and return it
# 3. use mapping function F to transform the input_image
# 4. save transformed image as <input_image_name>-histeq.png
# 5. plot the histogram of your transformed image (you may use the prohibited functions for this task)

# Href is the linear cumulative reference histogram
# nref is the number of pixels in the image

def print_usage():
    print("Usage: \n\tpython3 histmatch.py <input_image_name>\n")
    exit()

def create_Href():
    Href = np.zeros(256)
    for i in range(0,256):
        Href[i] = i * 1
    return Href

def main(argv):
    # print("hello world")
    if not(len(argv) == 1):
        print("Incorrect number of input arguments.")
        print_usage()

    # open image and check that it is grayscale
    try:
        img = cv2.imread(str(argv[0]),0)
        cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)  # normalize to between 0 and 255
    except IOError:
        print("Could not open image.")
        exit()
    h,w = img.shape
    # print("original image has "+str(c)+" channels")
    # if c == 3:
    #     print(c)
    #     print("Image is not grayscale.")
    #     exit()
    # nPixels = h*w

    # create linear cumulative histogram Href
    Href = create_Href()
    nref = 255

    # call 
    F = match_histograms(img, Href, nref)

    newhist = np.zeros(256)

    output_image = img.copy()
    height,width = output_image.shape
    for y in range(0,width):
        for x in range(0,height):
            pixelval = img[x,y]
            output_image[x,y] = F[pixelval]
            newhist[output_image[x,y]] += 1

    # plt.plot(newhist)
    # plt.show()


    cv2.imshow("title",np.concatenate((img,output_image[:,:]),axis=1))
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main(sys.argv[1:])