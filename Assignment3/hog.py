#!/usr/bin/env python3
# ECE 471/536: Assignment 3 submission template

"""
Geena Smith
V00835915
"""


#Using "as" nicknames a library so you don't have to use the full name
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse as ap
import pprint

from skimage.feature import hog as hg

pp = pprint.PrettyPrinter(indent=4)

#Prevents python3 from failing
TODO = None

#Epsilon 
EPS=1e-6

"""Extract Histogram of Gradient features

Parameters
----------
X : ndarray NxHxW array where N is the number of instances/images 
                              HxW is the image dimensions

Returns
    features : NxD narray contraining the histogram features (D = 2304) 
-------

"""
#20 marks: Histogram of Gradients
def hog(X):

    #2 marks: apply a Sobel filter for each gradient (x,y) of kernel size 1
    #         HINT: your gx/gy size should be NxHxW

    N,H,W = X.shape
    
    gx = np.array([cv2.Sobel(X[n,:,:],cv2.CV_32F,1,0,ksize=1) for n in range(0,N)],dtype=np.float32)
    gy = np.array([cv2.Sobel(X[n,:,:],cv2.CV_32F,0,1,ksize=1) for n in range(0,N)],dtype=np.float32)
    
    #2 marks: Calculate the mag and unsigned angle in range ( [0-180) degrees) for each pixel
    #         mag, ang should have size NxHxW

    mag = np.array([np.sqrt( np.power(gx[n,:,:],2) + np.power(gy[n,:,:],2) ) for n in range(0,N)],dtype=np.float32)
    ang = np.array([np.rad2deg( np.arctan( gy[n,:,:]/(gx[n,:,:] + 0.000001)) ) for n in range(0,N)],dtype=np.float32)
    ang[ang < 0] = 360 + ang[ang < 0]

    #1 mark: Split orientation matrix/tensor into 8x8 cells, flattened to 64
    #        HINT: matrix size should be Nx(number of cells)x(8x8)
    
    splitAng = split_into_cells(ang,8)

    #1 mark: Split magnitude matrix/tensor into 8x8 cells, flattened to 64
    #        HINT: matrix size should be Nx(number of cells)x(8*8)

    splitMag = split_into_cells(mag,8)

    #1 mark: create an array to hold the feature histogram for each 8x8 cell in a image
    #        HINT: the array should have 3 dimensions 

    _,numCells,numPixels = splitAng.shape
    featureHists = np.zeros((N,numCells,9))

    #Loop through and for each cell calculate the histogram of gradients
    #Don't worry if this is very slow 

    for n in range(0,N):
        for i in range(0,numCells):
            for j in range(0,numPixels):
                magVal = splitMag[n,i,j]
                angVal = splitAng[n,i,j]

    #1 mark: Find the two closest bins based on orientation of the gradient for pixel j in cell i

                bin1 = int(angVal / 20) % 9
                bin2 = (bin1+1) % 9


    #1 mark: Calculate the bin ratio (how much of the magnitude is added to each bin)
                ratio1 = 1-((angVal%20)/20)
                ratio2 = 1-ratio1
       
    #5 marks: add the magnitude contribution to each bin, based on orientation overlap with the bin (bin ratio)
    #         HINT: consider the edge cases for the bins
                featureHists[n,i,bin1] += ratio1 * magVal
                featureHists[n,i,bin2] += ratio2 * magVal

    #Normally, there is a window normalization step here, but we're going to ignore that.

    #1 mark: Reshape the histogram so that its NxD where N is the number of instances/images i
    #        and D is all the histograms per image concatenated into 1D vector

    features = np.array([featureHists[n,:,:].reshape((numCells*9)) for n in range(0,N)],dtype=np.float32)

    #return the histogram as your feature vector
    return features

#5 marks: Split the input matrix A into cells
def split_into_cells(A,cell_size=8):
    """Split ndarray into smaller array

    Parameters
    ----------
    A : ndarray of size NxHxW
    cell : tuple with (h,w) for cell size 

    Returns
    -------
    ndarray of size Nx(number of cells)x(cell_size*cell_size)
    """

    N,H,W = A.shape
    
    newH = int(H/cell_size)
    newW = int(W/cell_size)

    newArr = np.empty((N,newH*newW,cell_size*cell_size))

    for n in range(0,N):
        c = 0
        h = 0
        while h < H:
            w = 0
            while w < W:
                newArr[n,c,:] = A[n,h:(h+cell_size),w:(w+cell_size)].reshape((cell_size*cell_size))
                c += 1
                w += cell_size
            h += cell_size
    return newArr
