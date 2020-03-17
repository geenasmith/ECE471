from hog import split_into_cells

import numpy as np
if __name__ == '__main__':
    #Create two "images" that are 6x6 in size
    #First image only has values between 0-35
    A = np.arange(36).reshape((1, 6, 6))
    #Second image only has values between 36 - 71
    B = np.arange(36,36*2).reshape((1, 6, 6))
    #Comebine them to get 2x6x6 (a simple test case similar to the HOG input)
    C = np.concatenate((A,B))

    """
    Run your split_into_cells with a cell size of 3
    The output should be: 2x4x9
    """
    split = split_into_cells(C, cell_size=3)
    
    #Run two tests to verify the code is correct
    #Test 1: Make sure the output size is correct
    assert split.shape == (2,4,9) and "Split output is not the correct shape"
    #Test 2: Make sure it was split correctly by checking that the first image, first cell is correct
    assert np.array_equal(split[0][0], [ 0,1,2,6,7,8,12,13,14]) and "Split not done correctly, first cell isn't correct"
    
    #If you get this far, than your function is correct
    print("Split correct")
