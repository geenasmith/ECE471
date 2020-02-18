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

    g_k_1 = prev_pyr_level
    g_k_1_Height, g_k_1_Width, d = g_k_1.shape

    g_k_Height = int((g_k_1_Height-1)/2)+1
    g_k_Width = int((g_k_1_Width-1)/2)+1
    g_k_Depth = d
    g_k = np.zeros((g_k_Height,g_k_Width,g_k_Depth),np.uint8)

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
    # cv2.imshow('image',g_k)
    # cv2.waitKey(0)
    pyr_level = g_k



    # orig_height,orig_width,orig_depth = prev_pyr_level.shape

    # new_width = int(((orig_width-1)/2)+1)
    # new_height = int(((orig_height-1)/2)+1)

    # pyr_level = np.zeros((new_height,new_width,orig_depth),np.uint8)

    # prev_pyr_level = cv2.copyMakeBorder(prev_pyr_level,2,2,2,2,cv2.BORDER_REFLECT)  #add padding
    # orig_height,orig_width,orig_depth = prev_pyr_level.shape

    # # gl+1(i,j) = sum from m=-2,2 sum from n=-2,2 w(m,n)gl(2i-m,2j-n)

    # large_height_offset = int((orig_height-1)/2)
    # large_width_offset = int((orig_width-1)/2)

    # small_height_offset = int((new_height-1)/2)
    # small_width_offset = int((new_width-1)/2)

    # for i in range(-small_height_offset,small_height_offset+1):
    #     for j in range(-small_width_offset,small_width_offset+1):
    #         R_pixel_val = 0
    #         G_pixel_val = 0
    #         B_pixel_val = 0
    #         for m in range(-2,3):
    #             for n in range(-2,3):
    #                 R_pixel_val += kernel[m+2][n+2]*prev_pyr_level[(2*i)-(m)+large_height_offset][(2*j)-(n)+large_width_offset][0]
    #                 G_pixel_val += kernel[m+2][n+2]*prev_pyr_level[(2*i)-(m)+large_height_offset][(2*j)-(n)+large_width_offset][1]
    #                 B_pixel_val += kernel[m+2][n+2]*prev_pyr_level[(2*i)-(m)+large_height_offset][(2*j)-(n)+large_width_offset][2]
    #         pyr_level[i+small_height_offset][j+small_width_offset][0] = int(R_pixel_val)
    #         pyr_level[i+small_height_offset][j+small_width_offset][1] = int(G_pixel_val)
    #         pyr_level[i+small_height_offset][j+small_width_offset][2] = int(B_pixel_val)
    # # print(str(mn) + "," + str(mx))

    # cv2.imshow('image',pyr_level)
    # cv2.waitKey(0)

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

    g_k = np.zeros((size[0],size[1],size[2]),np.uint8)

    g_k_Height = size[0]
    g_k_Width = size[1]
    g_k_Depth = size[2]

    print(g_k_1.shape)
    print(g_k.shape)
    
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
    
    cv2.imshow('image',g_k)
    cv2.waitKey(0)
    pyr_level = g_k

    # expand(gl)=4 sum from m=-2,2 sum from n=-2,2 w(m,n)gl((i-m)/2,(j-n)/2)

    # pyr_level = np.zeros((size[0],size[1],size[2]),np.uint8)  #larger image

    # h,w,d = prev_pyr_level.shape  # smaller image

    # large_height_offset = int((size[0]-1)/2)
    # large_width_offset = int((size[1]-1)/2)

    # small_height_offset = int((h-1)/2)+2
    # small_width_offset = int((w-1)/2)+2

    # cv2.imshow('image',prev_pyr_level)
    # cv2.waitKey(0)

    # prev_pyr_level = cv2.copyMakeBorder(prev_pyr_level,2,2,2,2,cv2.BORDER_REFLECT)

    # for i in range(-large_height_offset,large_height_offset+1):
    #     for j in range(-large_width_offset,large_width_offset+1):
    #         R_pixel_val = 0
    #         G_pixel_val = 0
    #         B_pixel_val = 0
    #         for m in range(-2,3):
    #             for n in range(-2,3):
    #                 R_pixel_val += 4*kernel[m+2][n+2]*prev_pyr_level[int((i-m)/2)+small_height_offset][int((j-n)/2)+small_width_offset][0]
    #                 G_pixel_val += 4*kernel[m+2][n+2]*prev_pyr_level[int((i-m)/2)+small_height_offset][int((j-n)/2)+small_width_offset][1]
    #                 B_pixel_val += 4*kernel[m+2][n+2]*prev_pyr_level[int((i-m)/2)+small_height_offset][int((j-n)/2)+small_width_offset][2]
    #         pyr_level[i+large_height_offset][j+large_width_offset][0] = int(R_pixel_val)
    #         pyr_level[i+large_height_offset][j+large_width_offset][1] = int(G_pixel_val)
    #         pyr_level[i+large_height_offset][j+large_width_offset][2] = int(B_pixel_val)

    # cv2.imshow('image',pyr_level)
    # cv2.waitKey(0)
    # cv2.imwrite('test.jpg',pyr_level)
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

    prev_pyr_level = img
    gaussian_pyr = [prev_pyr_level]
    kernel = make_kernel(alpha)
    for N in range(0,num_levels-1):
        new_img = pyrDown(prev_pyr_level, kernel)
        gaussian_pyr.append(new_img)
        prev_pyr_level = new_img
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

    # L0 = g0 - Expand(g1)
    # for N gaussian images, will have N-1 laplace
    # get g0
    # expand g1
    # subtract (np.subtract)

    N = len(gaussian_pyr)
    laplacian_pyr = []

    kernel = make_kernel(alpha)

    for n in range(0,N-1):
        g_k = gaussian_pyr[n+1]
        g_k_1 = gaussian_pyr[n]
        size = g_k_1.shape
        g_k_exp = pyrUp(g_k, size, kernel)
        sub_img = np.subtract(g_k_1,g_k_exp)
        laplacian_pyr.append(sub_img)
        cv2.imshow('image',sub_img)
        cv2.waitKey(0)


        # prev_pyr_level = gaussian_pyr[n+1]  #expand this
        # size = gaussian_pyr[n].shape
        # expanded_img = pyrUp(prev_pyr_level, size, kernel)
        # sub_img = np.subtract(gaussian_pyr[n],expanded_img)
        # laplacian_pyr.append(sub_img)
        # cv2.imshow('image',sub_img)
        # cv2.waitKey(0)
    laplacian_pyr.append(gaussian_pyr[-1])
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

    # print(len(laplacianA))
    # print(len(laplacianB))
    # print(len(region_pyr))
    # print(laplacianA[-1].shape)
    # print(laplacianB[-1].shape)
    # print(region_pyr[0].shape)
    # print(type(region_pyr[0]))

    blended_pyr = []

    # cv2.imshow('image',blended_layer)
    # cv2.waitKey(0)

    for n in range(0,len(laplacianA)):
        # blended_layer = mask_pyr[-n-1]*laplacianA[n] + (1-mask_pyr[-n-1])*laplacianB[n]
        blended_layer = np.multiply(region_pyr[-n-1],laplacianA[n]) + np.multiply(np.subtract(1,region_pyr[-n-1]),laplacianB[n])
        blended_pyr.append(blended_layer)
        # cv2.imshow('image',blended_layer)
        # cv2.waitKey(0)

    # LS_l(i,j) = GR_l(i,j)*LA_l(i,j) + (1-GR_l(i,j))*LB_l(i,j)


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

    print(blended_pyr[0].shape)
    print(blended_pyr[-1].shape)

    kernel = make_kernel(alpha)

    recon_image = blended_pyr[0]

    # index n will need to be expanded n-1 times
    for n in range(1,len(blended_pyr)):
        layer = blended_pyr[n]
        for i in range(0,n):
            h,w,d = layer.shape
            size = [(2*h)-1,(2*w)-1,d]
            layer = pyrUp(layer, size, kernel)
        recon_image = recon_image + layer


    # pyrUp(prev_pyr_level, size, kernel)

    return recon_image 

