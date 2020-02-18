import cv2
import numpy as np
import matplotlib.pyplot as plt

#Total marks: 25
TODO = None


# https://www.programcreek.com/python/example/89373/cv2.filter2D apply convolution example

"""
    Create kernel w from alpha (see lecture notes)

    Args:
    alpha : float scaling value

    Returns:
    kernel : 2 dimension square kernel w 
    2 mark
"""
def make_kernel(alpha):
    TODO

    w_hat = np.array([[(1/4)-(alpha/2),(1/4),alpha,(1/4),(1/4)-(alpha/2)]])

    kernel = w_hat.transpose()*w_hat

    return kernel

""" 
    Perform the REDUCE function to create a Gaussian pyramid layer 

    Args:
    prev_pyr_level : the previous Gaussian layer
    kernel : The weight kernel for blurring

    Returns: current Gaussian layer 
    2 marks
"""
def pyrDown(prev_pyr_level, kernel):
    TODO
    # print(prev_pyr_level.shape)

    g_k_1 = prev_pyr_level
    g_k_1_Height, g_k_1_Width, g_k_1_Depth = g_k_1.shape

    g_k_Height = int((g_k_1_Height-1)/2)+1
    g_k_Width = int((g_k_1_Width-1)/2)+1
    g_k_Depth = int(g_k_1_Depth)
    g_k = np.zeros((g_k_Height,g_k_Width,g_k_Depth))

    # prev_pyr_level = cv2.copyMakeBorder(prev_pyr_level,2,2,2,2,cv2.BORDER_REFLECT)
    g_k_1 = cv2.copyMakeBorder(g_k_1,2,2,2,2,cv2.BORDER_REFLECT)

    # i in [0,C_k), j in [0,R_k)
    # g_k(i,j) = sum from m=-2,2 sum from n=-2,2 w(m,n)*g_k_1(2i+m,2j+n)

    for i in range(0,g_k_Height):
        for j in range(0,g_k_Width):
            R = 0
            G = 0
            B = 0
            for m in range(-2,3):
                for n in range(-2,3):
                    R += kernel[m+2][n+2]*g_k_1[(2*i)+m+2][(2*j)+n+2][0]
                    G += kernel[m+2][n+2]*g_k_1[(2*i)+m+2][(2*j)+n+2][1]
                    B += kernel[m+2][n+2]*g_k_1[(2*i)+m+2][(2*j)+n+2][2]
            g_k[i][j][0] = R
            g_k[i][j][1] = G
            g_k[i][j][2] = B
    # cv2.imshow('image',g_k_1)
    # cv2.waitKey(0)
    # cv2.imshow('image',g_k)
    # cv2.waitKey(0)
    pyr_level = g_k

    return pyr_level

""" 
    Perform the EXPAND function to create a Laplacian pyramid expanded layer 
    
    Args:
    prev_pyr_level : the previous Gaussian layer
    size   : Size of expanded layer
    kernel : The weight kernel for blurring

    Returns: current pyramid layer 
    2 mark
"""
def pyrUp(prev_pyr_level, size, kernel):
    TODO

    g_k_1 = prev_pyr_level
    g_k_1_Height, g_k_1_Width, g_k_1_Depth = g_k_1.shape

    g_k = np.zeros((size[0],size[1],size[2]))

    g_k_Height = size[0]
    g_k_Width = size[1]
    g_k_Depth = size[2]

    # print(g_k_1.shape)
    # print(g_k.shape)
    
    g_k_1 = cv2.copyMakeBorder(g_k_1,2,2,2,2,cv2.BORDER_REFLECT)

    for i in range(0,g_k_Height):
        for j in range(0,g_k_Width):
            R = 0
            G = 0
            B = 0
            for m in range(-2,3):
                for n in range(-2,3):
                    ind1 = (i-m)/2
                    ind2 = (j-n)/2
                    if ind1%1==0 and ind2%1==0:
                        R += 4 * kernel[m+2][n+2] * g_k_1[int(ind1)+2][int(ind2)+2][0]
                        G += 4 * kernel[m+2][n+2] * g_k_1[int(ind1)+2][int(ind2)+2][1]
                        B += 4 * kernel[m+2][n+2] * g_k_1[int(ind1)+2][int(ind2)+2][2]
            g_k[i][j][0] = R
            g_k[i][j][1] = G
            g_k[i][j][2] = B
    
    # cv2.imshow('image',g_k)
    # cv2.waitKey(0)
    pyr_level = g_k

    # expand(gl)=4 sum from m=-2,2 sum from n=-2,2 w(m,n)gl((i-m)/2,(j-n)/2)

    return pyr_level 

""" 
    This function will create a Gaussian pyramid of N levels
    use alpha to construct a kernel w

    Args:
    img: square image (2n+1,2n+1, 3)
    num_levels: height or number of pyramid layers
    alpha = See lecture notes, scaling value for kernel w

    Returns: list of gaussian pyramid layers
    5 marks
"""
def gaussian_pyramid(img, num_levels, alpha):
    TODO

    kernel = make_kernel(alpha)

    gaussian_pyr = [img]

    # prev_pyr_level = img

    for N in range(1,num_levels):

        gaussian_pyr.append(pyrDown(gaussian_pyr[-1],kernel))

    return gaussian_pyr

""" 
    This function will create a Laplacian pyramid from the
    Gaussian pyramid by using EXPAND.

    Args:
    gaussian_pyr : list of Gaussian layers (from gaussian_pyr)
    alpha : See lecture notes, scaling value for kernel w

    Returns: list of laplacian pyramid layers
    6 marks
"""
def laplacian_pyramid(gaussian_pyr, alpha):
    TODO

    kernel = make_kernel(alpha)

    laplacian_pyr = []
    N = len(gaussian_pyr)

    for n in range(0,N):
        if n+1 == N:
            laplacian_pyr.append(gaussian_pyr[n])
        else:
            exp_image = pyrUp(gaussian_pyr[n+1],gaussian_pyr[n].shape,kernel)
            laplacian_pyr.append(np.subtract(gaussian_pyr[n],exp_image))

    # laplacian_pyr.append(gaussian_pyr[-1])
    return laplacian_pyr

""" This function will blend the two pyramids and the region
    mask.

    Args:
    laplacianA : laplacian pyramid for left image
    laplacianB : laplacian pyramid for right image
    mask_pyr   : gaussian pyramid for region mask

    Return
    blended_pyr : list of blended pyramid layers
    2 marks
"""
def blend(laplacianA,laplacianB,region_pyr):
    TODO

    blended_pyr = []
    n = len(region_pyr)
    for l in range(0,len(region_pyr)):
        blended_layer = np.multiply(region_pyr[n-l-1],laplacianB[l]) + np.multiply( np.subtract(1,region_pyr[n-l-1]) , laplacianA[l] )
        blended_pyr.append(blended_layer)


    # cv2.imshow('image',laplacianA[1])
    # cv2.waitKey(0)
    # cv2.imshow('image',laplacianB[1])
    # cv2.waitKey(0)
    # cv2.imshow('image',region_pyr[1])
    # cv2.waitKey(0)
    # cv2.imshow('image',blended_pyr[1])
    # cv2.waitKey(0)

    return blended_pyr

""" 
    This function will reconstruct/collapse the laplacian pyramid into a single image
    
    Args:
    blended_pyr : list of blended laplacian layers (from blend)
    alpha : See lecture notes, scaling value for kernel w

    Returns: final reconstructed image 
    6 marks
"""
def reconstruct(blended_pyr, alpha):
    TODO

    kernel = make_kernel(alpha)

    # start with smallest layer, expand it, then add to next smallest layer, repeat
    n = len(blended_pyr)

    recon_image = np.array(blended_pyr[0].shape,np.uint8)

    for l in range(n-1,0,-1):
        tmp = pyrUp(blended_pyr[l],blended_pyr[l-1].shape,kernel)
        blended_pyr[l-1] += tmp
    
    recon_image = blended_pyr[0]

    return recon_image 

