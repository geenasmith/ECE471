import numpy as np
import cv2
import matplotlib.pyplot as plt

from a2 import *
   
#Constants
num_levels = 5 #Number of pyramid levels
alpha = 0.375  #Alpha value for w

# 1. Load the two images
img1 = cv2.imread('../images/orange.jpg').astype(np.float32)
img2 = cv2.imread('../images/apple.jpg').astype(np.float32)

# 1. Create the mask to blend the two images
region = np.zeros(img1.shape, dtype='float32')
region[:,:img1.shape[1]//2,:] = (1,1,1)

# 2. Create the Gaussian pyramids 
gaussian_pyr_1 = gaussian_pyramid(img1, num_levels, alpha)
gaussian_pyr_2 = gaussian_pyramid(img2, num_levels, alpha)
region_pyr       = gaussian_pyramid(region, num_levels, 0.4)
region_pyr.reverse() 

# 3. Create the Laplacian pyramids
laplacian_pyr_1 = laplacian_pyramid(gaussian_pyr_1, alpha)
laplacian_pyr_2 = laplacian_pyramid(gaussian_pyr_2, alpha)

# 4. Blend the pyramids 
blended_pyr = blend(laplacian_pyr_1,laplacian_pyr_2, region_pyr)

# # 5. Reconstruct the images and return the final layer
final  = reconstruct(blended_pyr, alpha)
final  = np.uint8(np.clip(final, 0, 255)) #Force RGB range and datatype

# # 7. Save the final image to the disk
cv2.imwrite('blended.jpg',final)
