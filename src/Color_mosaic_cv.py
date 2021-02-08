import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread('../images/lights.jpg', 1)
#img.shape -> check the dimensions of the array
#img[i] -> represents the ith row of an image
#img[i][j] -> represents the RGB values at ith row and jth column
#         -> returns [R G B] array 

# img[3][4][0] = 4
# print(img[3][4])

height, width, channels = img.shape
print(img.shape) 

#pixels are in GBR (reverse RGB) order
#creating the bayer's pattern color mosaic
for i in range(0, height-1, 2):
    for j in range(0, width-1, 2):
        #red pixel
        img[i][j][0] = 0
        img[i][j][1] = 0
        #green pixel
        img[i][j+1][0] = 0 
        img[i][j+1][2] = 0 
        #green pixel
        img[i+1][j][0] = 0
        img[i+1][j][2] = 0
        #blue pixel
        img[i+1][j+1][1] = 0 
        img[i+1][j+1][2] = 0 

#creating bayer's pattern for the LAST extra column (if there's an odd number of columns)
if width%2 != 0:
    for i in range(0, height-1, 2):
        #red pixel (for the last column)
        img[i][width-1][0] = 0
        img[i][width-1][1] = 0
        #green pixel (for the last column)
        img[i+1][width-1][0] = 0
        img[i+1][width-1][2] = 0

#creating bayer's pattern for the LAST extra row (if there's an odd number of rows)
if height%2 != 0:
    for i in range(0, width-1, 2):
        #red pixel (for the last row)
        img[height-1][i][0] = 0
        img[height-1][i][1] = 0
        #green pixel (for the last row)
        img[height-1][i+1][0] = 0
        img[height-1][i+1][2] = 0

#cv2.imshow("image",img)
#cv2.waitKey(0)

cv2.imwrite('../images/lights.png', img)