# ECE 471 - Assingment 1
# Geena Smith
# V00835915

import sys, os, cv2, matplotlib.pyplot as plt, numpy as np
from match_histograms import match_histograms

def print_usage():
    print("Usage: \n\tpython3 histmatch.py <input_image_name>\n")
    exit()

def main(argv):
    if not(len(argv) == 1):
        print("Incorrect number of input arguments.")
        print_usage()

    # open image and check that it is grayscale
    try:
        input_image = cv2.imread(str(argv[0]),0)
        cv2.normalize(input_image, input_image, 0, 255, cv2.NORM_MINMAX)  # normalize to between 0 and 255
    except IOError:
        print("Could not open image.")
        exit()

    # get number of pixels in the image
    height,width = input_image.shape

    # create linear cumulative histogram Href and define nref
    nref = height*width
    Href = [i*nref/255 for i in range(0,256)]

    # call match_histograms
    F = match_histograms(input_image, Href, nref)
    F = F.astype(np.uint8)  # convert to unsigned integer

    output_image = input_image.copy()
    for y in range(0,width):
        for x in range(0,height):
            output_image[x,y] = F[input_image[x,y]]

    # Create the new output file name and write the image
    # https://stackoverflow.com/questions/678236/how-to-get-the-filename-without-the-extension-from-a-path-in-python
    outputFileName = os.path.splitext(str(argv[0]))[0] + '-histeq.png'
    cv2.imwrite(outputFileName,output_image)

    # Plot the original and new histograms (non-cumulative)
    fig, ax = plt.subplots(1,2)
    ax[0].hist(input_image.ravel(),256,[0,256])
    ax[0].set_title('Original Histogram')
    ax[1].hist(output_image.ravel(),256)
    ax[1].set_title('Transformed Histogram')
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])