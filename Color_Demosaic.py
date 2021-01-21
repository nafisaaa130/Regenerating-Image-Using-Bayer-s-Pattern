import os
import imageio
import numpy as np

im = imageio.imread('./bird.jpg')
#im.shape -> check the dimensions of the array
#im[i] -> represents the ith row of an image
#im[i][j] -> represents the RGB values at ith row and jth column
#         -> returns [R G B] array 
 
print("Dimensions of the array (Numpy): ", im.shape)
print(im[0][0])

#dimensions of the image
totalRows, totalColumns = im.shape[0], im.shape[1]

#look at every other line for G and R, starting at the first line
#look at every other line for B and G, starting at the second line
pixel = im[0][0]
#at one column, different rows
print(pixel)

#i = row and j = column

#MIGHT need to individually go through each RGB value and have a mathematical formula for decreasing the values

for i in range(0,totalRows-1,2):
    for j in range(0, totalColumns-1, 2):
        #line containing green and red pixels
            #im[i][j] = np.subtract(im[i][j], [200, 0, 200]) #green pixel    
            im[i][j][0] =
            im[i][j][1] =
            im[i][j][2] =

            #im[i][j+1] = np.subtract(im[i][j+1], [0, 200, 200]) #red pixel
            im[i][j+1][0] =
            im[i][j+1][1] =
            im[i][j+1][2] =

        #line containing blue and green pixels
            #im[i+1][j] = np.subtract(im[i+1][j], [200, 200, 0]) #blue pixel
            im[i+1][j][0] =
            im[i+1][j][1] =
            im[i+1][j][2] =

            #im[i+1][j+1] = np.subtract(im[i+1][j+1], [200, 0, 200])#green pixel
            im[i+1][j+1][0] =
            im[i+1][j+1][1] =
            im[i+1][j+1][2] =

imageio.imwrite('./bird_modified.jpg', im[:, :, 0])