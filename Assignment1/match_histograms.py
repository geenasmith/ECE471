# ECE 471 - Assingment 1
# Geena Smith
# V00835915

import sys, os, cv2, matplotlib.pyplot as plt, numpy as np

# Algorithm implemented from that given by Amanda Dash in ECE 471 Lecture 03 Slide 62
def match_histograms(input_image, Href, nref):
    height,width = input_image.shape
    n = width*height
    r = n/nref
    h = np.zeros(256)  #histogram of input_image
    for y in range(0,width):
        for x in range(0,height):
            pixelval = input_image[x,y]
            h[pixelval] = h[pixelval] + 1

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
