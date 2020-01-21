# ECE 471 - Assingment 1
# Geena Smith
# V00835915

# matchHistograms(pixels,width,height,Href,nref)
#   n <-- width*height
#   r <-- n/nref
#   h <-- (non)-cumulative histogram of the pixels array
#   F <-- integer array of size 256, initialized to 0
#   initialize i,j,c to 0
#   while i < 256 do
#       if c <= r*Href[j] then
#           c <-- c+h[i]
#           F[i] <-- j
#           i ++
#       else
#           j++
#       end if
#   end while
# return F
# end procedure

import sys, os, cv2, matplotlib.pyplot as plt, numpy as np

def match_histograms(input_image, Href, nref):
    height,width = input_image.shape
    n = width*height
    r = n/nref
    h = np.zeros(256)  #histogram of input_image
    for y in range(0,width):
        for x in range(0,height):
            pixelval = input_image[x,y]
            h[pixelval] = h[pixelval] + 1
    # plt.plot(h)
    # plt.show()

    F = np.zeros(256)
    i = 0
    j = 0
    c = 0

    while i < 256:
        if c <= (r*Href[j]):
            c = c+h[i]
            F[i] = j
            i+=1
        else:
            j += 1
    return F
