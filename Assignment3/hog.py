#!/usr/bin/env python3
# ECE 471/536: Assignment 3 submission template


#Using "as" nicknames a library so you don't have to use the full name
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse as ap
import pprint

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

    gx = np.empty_like(X)
    gy = np.empty_like(X)

    for n in range(0,N):
        gx[n,:,:] = cv2.Sobel(X[n,:,:],cv2.CV_64F,1,0,ksize=1)
        gy[n,:,:] = cv2.Sobel(X[n,:,:],cv2.CV_64F,0,1,ksize=1)


    # DELETE THIS SECTION
    # sobelx = cv2.Sobel(X[0,:,:],cv2.CV_64F,1,0,ksize=1)
    # sobely = cv2.Sobel(X[0,:,:],cv2.CV_64F,0,1,ksize=1)
    # plt.subplot(2,2,1),plt.imshow(X[0,:,:],cmap = 'gray')
    # plt.title('Original'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2,2,2),plt.imshow(sobelx,cmap = 'gray')
    # plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2,2,3),plt.imshow(sobely,cmap = 'gray')
    # plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    # plt.show()
    # exit()

    
    #2 marks: Calculate the mag and unsigned angle in range ( [0-180) degrees) for each pixel
    #         mag, ang should have size NxHxW
    
    mag = np.empty_like(X)
    ang = np.empty_like(X)

    for n in range(0,N):
        mag[n,:,:] = np.sqrt( np.power(gx[n,:,:],2) + np.power(gy[n,:,:],2) )  # sqrt(gx^2 + gy^2)
        ang[n,:,:] = np.rad2deg( np.arctan( gy[n,:,:]/(gx[n,:,:] + 0.000001)) ) % 180  # add small value as not to divide by 0
    


    exit()


    #1 mark: Split orientation matrix/tensor into 8x8 cells
    #        HINT: matrix size should be Nx(number of cells)x8x8
    
    




    #1 mark: Split magnitude matrix/tensor into 8x8 cells, flattened to 64
    #        HINT: matrix size should be Nx(number of cells)x(8*8)
    

    #1 mark: create an array to hold the feature histogram for each 8x8 cell in a image
    #        HINT: the array should have 3 dimensions 
    

    #Loop through and for each cell calculate the histogram of gradients
    #Don't worry if this is very slow 
   
    #1 mark: Find the two closest bins based on orientation of the gradient for pixel j in cell i

    #1 mark: Calculate the bin ratio (how much of the magnitude is added to each bin)

       
    #5 marks: add the magnitude contribution to each bin, based on orientation overlap with the bin (bin ratio)
    #         HINT: consider the edge cases for the bins
   
    #Normally, there is a window normalization step here, but we're going to ignore that.

    #1 mark: Reshape the histogram so that its NxD where N is the number of instances/images i
    #        and D is all the histograms per image concatenated into 1D vector

    #return the histogram as your feature vector
    pass

#5 marks: Split the input matrix A into cells
def split_into_cells(A,cell_size=8):
    """Split ndarray into smaller array

    Parameters
    ----------
    A : ndarray of size NxHxW
    cell : tuple with (h,w) for cell size 

    Returns
    -------
    ndarray of size Nx(cell_h*cell_w)x(cell_h*cell_w)
    """
    pass
