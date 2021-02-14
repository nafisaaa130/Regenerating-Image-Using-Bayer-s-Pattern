import cv2
import numpy as np
import matplotlib.pyplot as plt

def createColorMosaic(inputfile, bayerFile):
    img=cv2.imread(inputfile, 1)
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

    #creating bayer's pattern for the LAST extra row (if there's an odd number of is)
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

    cv2.imwrite(bayerFile, img)
#using edge-directed interpolation and nearest neighbor replication
def colorDemosaic(bayerFile):
    img=cv2.imread(bayerFile, 1)
    #img.shape -> check the dimensions of the array
    #img[i] -> represents the ith row of an image
    #img[i][j] -> represents the RGB values at ith row and jth column
    #         -> returns [R G B] array 

    # img[3][4][0] = 4
    # print(img[3][4])

    height, width, channels = img.shape
    print(img.shape) 

    #finalImage = np.zeros()

    rows = height
    columns = width

    if height%6 != 0:
        rows = (height//6)*6
    if width%6 !=0:
        columns = (width//6)*6

    #generating green pixels from red pixels using adaptive edge-directional interpolation
    for i in range(0, rows, 2):
        for j in range(0, columns, 2):
            #if there isn't pixel info on the next right or bottom edge, mirror
            if j+4 == rows-1 and i+2<rows:
                img[i+2][j][1] = img[i+2][j-1][1] #pixel from the left column

            elif i+4 == columns-1 and j+2 < columns:
                img[i][j+2][1] = img[i-1][j+2][1] #pixel from the previous row

            #pixel has information from all four edges
            if j+4 < width and i+4 < height:
                horizontal_grad = abs(int(img[i+2][j][2]) - (2*int(img[i+2][j+2][2])) + int(img[i+2][j+4][2])) + abs(int(img[i+2][j+3][1]) - int(img[i+2][j+1][1]))
                vertical_grad = abs(int(img[i][j+2][2]) - (2*int(img[i+2][j+2][2])) + int(img[i+4][j+2][2])) + abs(int(img[i+3][j+2][1]) - int(img[i+1][j+2][1]))
                
                if horizontal_grad < vertical_grad:
                    img[i+2][j+2][1] = ((int(img[i+2][j+1][1])+int(img[i+2][j+3][1]))/2) - ((int(img[i+2][j][2]) - (2*int(img[i+2][j+2][2])) + int(img[i+2][j+4][2]))/2)
                elif horizontal_grad > vertical_grad:
                    img[i+2][j+2][1] = ((int(img[i+1][j+2][1])+int(img[i+3][j+2][1]))/2) - ((int(img[i][j+2][2]) - (2*int(img[i+2][j+2][2])) + int(img[i+4][j+2][2]))/2)
                else:
                    img[i+2][j+2][1] = ((int(img[i+2][j+1][1])+int(img[i+2][j+3][1])+int(img[i+1][j+2][1])+int(img[i+3][j+2][1]))/4) - ((int(img[i][j+2][2])+int(img[i+2][j][2]) - (4*int(img[i+2][j+2][2])) +int(img[i+2][j+4][2])+int(img[i+4][j+2][2]))/4)
                
                # #filling in any missing green pixel values, with the average of other green pixels
                # if img[i+2][j+2][1] == 0:
                #     img[i+2][j+2][1] = (int(img[i+2][j+1][1]) + int(img[i+2][j+3][1]) + int(img[i+1][j+2][1]) + int(img[i+3][j+2][1]))/4

    #generating green pixels from blue pixels using adaptive edge-directional interpolation
    for i in range(1, rows+1, 2):
        for j in range(1, columns+1, 2):
            #if there isn't pixel info on the next right or bottom edge, mirror
            if j+4 == columns and i+2 < rows+1:
                img[i+2][j][1] = img[i+2][j-1][1] #pixel from the left column
                img[i+1][j+1][1] = img[i+1][j][1]

            elif i+4 == columns and j+2 < columns+1:
                img[i][j+2][1] = img[i-1][j+2][1] #pixel from the previous row
                img[i][j][1] = img[i-1][j][1]

            #pixel has information from all four edges
            if j+4 < width and i+4 < height:
                horizontal_grad = abs(int(img[i+2][j][0]) - (2*int(img[i+2][j+2][0])) + int(img[i+2][j+4][0])) + abs(int(img[i+2][j+3][1]) - int(img[i+2][j+1][1]))
                vertical_grad = abs(int(img[i][j+2][0]) - (2*int(img[i+2][j+2][0])) + int(img[i+4][j+2][0])) + abs(int(img[i+3][j+2][1]) - int(img[i+1][j+2][1]))
                
                if horizontal_grad < vertical_grad:
                    img[i+2][j+2][1] = ((int(img[i+2][j+1][1])+int(img[i+2][j+3][1]))/2) - ((int(img[i+2][j][0]) - (2*int(img[i+2][j+2][0])) + int(img[i+2][j+4][0]))/2)
                elif horizontal_grad > vertical_grad:
                    img[i+2][j+2][1] = ((int(img[i+1][j+2][1])+int(img[i+3][j+2][1]))/2) - ((int(img[i][j+2][0]) - (2*int(img[i+2][j+2][0])) + int(img[i+4][j+2][0]))/2)
                else:
                    img[i+2][j+2][1] = ((int(img[i+2][j+1][1])+int(img[i+2][j+3][1])+int(img[i+1][j+2][1])+int(img[i+3][j+2][1]))/4) - ((int(img[i][j+2][0])+int(img[i+2][j][0]) - (4*int(img[i+2][j+2][0])) +int(img[i+2][j+4][0])+int(img[i+4][j+2][0]))/4)
                
    #mirror the green pixels for the first 2 rows and 2 columns at the beginning - nearest neighbor replication
    #for the 2 rows
    for i in range(0, 2, 2):
        for j in range(0, columns, 2):
            img[i][j][1] = img[i][j+1][1]
            img[i+1][j+1][1] = img[i+1][j][1]
    
    #for the 2 columns
    for i in range(2, rows, 2):
        for j in range(0, 2, 2):
            img[i][j][1] = img[i][j+1][1]
            img[i+1][j+1][1] = img[i+1][j][1]

    #mirror the edges (green pixels) at the end of the picture (the last rows and columns)- nearest neighbor replication
    #extra rows
    if height%6 !=0:
        row_toggle = 0
        for i in range(rows, height):
            for j in range(0, width, 2):
                if j+1 < width-1:
                    if row_toggle == 0:
                        img[i][j][1] = img[i][j-1][1]
                    if row_toggle == 1:
                        img[i][j+1][1] = img[i][j][1]
            if row_toggle == 0:
                row_toggle = 1
            else:
                row_toggle = 0
    
    # #extra columns
    if width%6 !=0:
        col_toggle = 0
        for j in range(columns, width):
            for i in range(0, rows, 2):
                if i+1 < height-1:
                    if col_toggle == 0:
                        img[i][j][1] = img[i][j-1][1]
                    if col_toggle == 1:
                        img[i+1][j][1] = img[i+1][j-1][1]
            if col_toggle == 0:
                col_toggle = 1
            else:
                col_toggle = 0

    #generating red pixels
    rows = height
    columns = width
    #for 4 by 4 window -> using 3x3 window to calculate the interpolate the red pixels
    if height%4 != 0:
        rows = (height//4)*4
    if width%4 != 0:
        columns = (width//4)*4
    #generating red pixels using adaptive edge-directional interpolation
    for i in range(0, rows, 2):
        for j in range(0, columns, 2):
            if i+1 == height-1 and j+1 == width-1:
                img[i][j+1][2] = img[i][j][2]
                img[i+1][j][2] = img[i][j][2]
                img[i+1][j+1][2] = img[i][j+1][2]
                break
            elif i+1 == height-1 and j+1 < width:
                img[i+1][j][2] = img[i][j][2]
                img[i+1][j+1][2] = img[i][j+1][2]
                img[i+1][j+2][2] = img[i][j+2][2]
                break
            elif j+1 == width-1 and i+1 < height:
                img[i][j+1][2] = img[i][j][2]
                img[i+1][j+1][2] = img[i+1][j][2]
                img[i+2][j+1][2] = img[i+2][j][2]
                break

            if i+2 < rows-1 and j+2 < columns-1:
                img[i][j+1][2] = ((int(img[i][j][2])+int(img[i][j+2][2]))/2) - ((int(img[i][j][1]) - (2*int(img[i][j+1][1])) + int(img[i][j+2][1]))/2)
                img[i+1][j][2] = ((int(img[i][j][2])+int(img[i+2][j][2]))/2) - ((int(img[i][j][1]) - (2*int(img[i+1][j][1])) + int(img[i+2][j][1]))/2)
                img[i+1][j+2][2] = ((int(img[i][j+2][2])+int(img[i+2][j+2][2]))/2) - ((int(img[i][j+2][1]) - (2*int(img[i+1][j+2][1])) + int(img[i+2][j+2][1]))/2)
                img[i+2][j+1][2] = ((int(img[i+2][j][2])+int(img[i+2][j+2][2]))/2) - ((int(img[i+2][j][1]) - (2*int(img[i+2][j+1][1])) + int(img[i+2][j+2][1]))/2)

                horizontal_red = abs(int(img[i][j+2][1]) - (2*int(img[i+1][j+1][1])) + int(img[i+2][j][1])) + abs(int(img[i+2][j][2]) - int(img[i][j+2][2]))
                vertical_red = abs((int(img[i][j][1]) - (2*int(img[i+1][j+1][1])) + int(img[i+2][j+2][1]))) + abs(int(img[i+2][j+2][2]) - int(img[i][j][2]))

                if horizontal_red < vertical_red:
                    img[i+1][j+1][2] =  ((int(img[i][j+2][2])+int(img[i+2][j][2]))/2) - ((int(img[i][j+2][1]) - (2*int(img[i+1][j+1][1])) + int(img[i+2][j][1]))/2)

                elif horizontal_red > vertical_red:
                    img[i+1][j+1][2] =  ((int(img[i][j][2])+int(img[i+2][j+2][2]))/2) - ((int(img[i][j][1]) - (2*int(img[i+1][j+1][1])) + int(img[i+2][j+2][1]))/2)

                else:
                    img[i+1][j+1][2] =  ((int(img[i][j][2])+int(img[i][j+2][2]) + int(img[i+2][j][2]) + int(img[i+2][j+2][2]))/4) - ((int(img[i][j][1]) + int(img[i][j+2][1]) - (4*int(img[i+1][j+1][1])) + int(img[i+2][j][1]) + int(img[i+2][j+2][1]))/4)
    
    #mirror the edges (red pixels) at the end of the picture (the last rows and columns)- nearest neighbor replication
    #extra rows
    if height%4 !=0:
        row_toggle = 0
        for i in range(rows, height):
            for j in range(0, width, 2):
                if j+1 < width-1:
                    if row_toggle == 0:
                        img[i][j+1][2] = img[i-1][j+1][2]
                    if row_toggle == 1:
                        img[i][j][2] = img[i-1][j][2]
                        img[i][j+1][2] = img[i-1][j+1][2]
            if row_toggle == 0:
                row_toggle = 1
            else:
                row_toggle = 0
    
    # #extra columns
    if width%4 !=0:
        col_toggle = 0
        for j in range(columns, width):
            for i in range(0, rows, 2):
                if i+1 < height-1:
                    if col_toggle == 0:
                        img[i+1][j][2] = img[i+1][j-1][2]
                    if col_toggle == 1:
                        img[i][j][2] = img[i][j-1][2]
                        img[i+1][j][2] = img[i+1][j-1][2]
            if col_toggle == 0:
                col_toggle = 1
            else:
                col_toggle = 0
    
    
    rows = height
    columns = width
    #for 4 by 4 window -> using 3x3 window to calculate the interpolate the blue pixels
    if height%4 != 0:
        rows = (height//4)*4
    if width%4 != 0:
        columns = (width//4)*4
    #generating blue pixels using adaptive edge-directional interpolation
    for i in range(1, rows+1, 2):
        for j in range(1, columns+1, 2):
            if i+1 == height-1 and j+1 == width-1:
                img[i][j+1][0] = img[i][j][0]
                img[i+1][j][0] = img[i][j][0]
                img[i+1][j+1][0] = img[i][j+1][0]
                break
            elif i+1 == height-1 and j+1 < width:
                img[i+1][j][0] = img[i][j][0]
                img[i+1][j][0] = img[i][j][0]
                img[i+1][j+1][0] = img[i][j+1][0]
                break
            elif j+1 == width-1 and i+1 < height:
                img[i][j+1][0] = img[i][j][0]
                img[i+1][j][0] = img[i+1][j-1][0]
                img[i+1][j+1][0] = img[i+1][j][0]
                break

            if i+2 < rows and j+2 < columns:
                img[i][j+1][0] = ((int(img[i][j][0])+int(img[i][j+2][0]))/2) - ((int(img[i][j][1]) - (2*int(img[i][j+1][1])) + int(img[i][j+2][1]))/2)
                img[i+1][j][0] = ((int(img[i][j][0])+int(img[i+2][j][0]))/2) - ((int(img[i][j][1]) - (2*int(img[i+1][j][1])) + int(img[i+2][j][1]))/2)
                img[i+1][j+2][0] = ((int(img[i][j+2][0])+int(img[i+2][j+2][0]))/2) - ((int(img[i][j+2][1]) - (2*int(img[i+1][j+2][1])) + int(img[i+2][j+2][1]))/2)
                img[i+2][j+1][0] = ((int(img[i+2][j][0])+int(img[i+2][j+2][0]))/2) - ((int(img[i+2][j][1]) - (2*int(img[i+2][j+1][1])) + int(img[i+2][j+2][1]))/2)

                horizontal_red = abs(int(img[i][j+2][1]) - (2*int(img[i+1][j+1][1])) + int(img[i+2][j][1])) + abs(int(img[i+2][j][0]) - int(img[i][j+2][0]))
                vertical_red = abs((int(img[i][j][1]) - (2*int(img[i+1][j+1][1])) + int(img[i+2][j+2][1]))) + abs(int(img[i+2][j+2][0]) - int(img[i][j][0]))

                if horizontal_red < vertical_red:
                    img[i+1][j+1][0] =  ((int(img[i][j+2][0])+int(img[i+2][j][0]))/2) - ((int(img[i][j+2][1]) - (2*int(img[i+1][j+1][1])) + int(img[i+2][j][1]))/2)

                elif horizontal_red > vertical_red:
                    img[i+1][j+1][0] =  ((int(img[i][j][0])+int(img[i+2][j+2][0]))/2) - ((int(img[i][j][1]) - (2*int(img[i+1][j+1][1])) + int(img[i+2][j+2][1]))/2)

                else:
                    img[i+1][j+1][0] =  ((int(img[i][j][0])+int(img[i][j+2][0]) + int(img[i+2][j][0]) + int(img[i+2][j+2][0]))/4) - ((int(img[i][j][1]) + int(img[i][j+2][1]) - (4*int(img[i+1][j+1][1])) + int(img[i+2][j][1]) + int(img[i+2][j+2][1]))/4)


    #mirror the edges (blue pixels) at the end of the picture (the last rows and columns)- nearest neighbor replication
    #extra rows
    if height%4 !=0:
        row_toggle = 0
        for i in range(rows+1, height):
            for j in range(0, width, 2):
                if j+1 < width-1:
                    if row_toggle == 0:
                        img[i][j][0] = img[i-1][j][0]
                    if row_toggle == 1:
                        img[i][j][0] = img[i-1][j][0]
                        img[i][j+1][0] = img[i-1][j+1][0]
            if row_toggle == 0:
                row_toggle = 1
            else:
                row_toggle = 0
    
    #extra columns
    if width%4 !=0:
        col_toggle = 0
        for j in range(columns+1, width):
            for i in range(0, rows+1, 2):
                if i+1 < height-1:
                    if col_toggle == 0:
                        img[i][j][0] = img[i][j-1][0]
                    if col_toggle == 1:
                        img[i][j][0] = img[i][j-1][0]
                        img[i+1][j][0] = img[i+1][j-1][0]
            if col_toggle == 0:
                col_toggle = 1
            else:
                col_toggle = 0
    
    #fill in blue pixels for first row - nearest neighbor replication
    img[1][0][0] = img[1][1][0]
    for j in range(0, width, 2):
        img[0][j][0] = img[1][j][0]
        if j+1 < width-1:
            img[0][j+1][0] = img[1][j+1][0]
    #fill in blue pixels for first column - nearest neighbor replication
    for i in range(2, height, 2):
        img[i][0][0] = img[i][1][0]
        if i+1 < height:
            img[i+1][0][0] = img[i+1][1][0]

    return img

#calculating the mean square error
def MSE(orig_img,interpol_img):
    orig_matrix = cv2.imread(orig_img)
    interpol_matrix = cv2.imread(interpol_img)

    squared_matrix = np.square(np.subtract(orig_matrix,interpol_matrix))
    average = squared_matrix.mean()

    return average

if __name__ == "__main__":
    inputFile = '../images/lights.jpg'
    #KEEP as png file for the jor mosaic to be generated properly
    
    # inputFile = '../images/lights.jpg'
    bayerFile = '../images/bayer.png'
    outputFile = '../images/interpolated.png'

    #creating color mosaic
    createColorMosaic(inputFile, bayerFile)
    print("Color mosaic of the image has been created")

    #interpolating the color mosaic
    image1 = colorDemosaic(bayerFile)
    #writing the image array to file
    cv2.imwrite(outputFile, image1)
    #calculating the MSE
    MSE = MSE(inputFile,outputFile)
    print(MSE)
