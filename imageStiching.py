#===============================================
# NAME: Joshua Preuss
# DATE: 5/7/2024
# CLASS: CSC 340-001
#===============================================

import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
import math

def matrix_multiply(matrix1, matrix2):
    """
    Multiply 2 nxn matricies
    Param: matrix1, matrix2 (matricies as lists of lists)
    Return: new_matrix, the resulting matrix when multiplication is performed
    """
    # Tests if matricies are compatible for multiplication
    if len(matrix1[0]) != len(matrix2):
        # raises value error if matricies are incompatable
        raise ValueError("Matricies cannot be multiplied. # of columns in matrix 1 must match # of rows in matrix 2")
    # initalizes a new matrix to put the values in
    new_matrix = [[0 for i in range(len(matrix2[0]))] for i in range(len(matrix1))]

    # loops through and multiples the matricies
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                new_matrix[i][j] += matrix1[i][k] * matrix2[k][j]
                
    return new_matrix


def main():
    # Load images
    im1Name = 'car1.jpg'
    im2Name = 'car2.jpg'
    img1 = cv2.imread(im1Name,0) # queryImage
    img2 = cv2.imread(im2Name,0) # trainImage
    numRows1 = img1.shape[0]
    numCols1 = img1.shape[1]
    numRows2 = img2.shape[0]
    numCols2 = img2.shape[1]
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2) #NOTE: 'None' parameter has to be added (not in documentation)

    plt.imshow(img3),plt.show()


    pts1 = np.zeros((len(good),2), np.float32)
    pts2 = np.zeros((len(good),2), np.float32)
    for m in range(len(good)):
            pts1[m] = kp1[good[m][0].queryIdx].pt
            pts2[m] = kp2[good[m][0].trainIdx].pt

    opencvH, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    print("H matrix estimated by OpenCV (for comparison):\n", opencvH)



    #### Get dimensions of pano
    corners = [[[0],[0],[1]],[[numCols1],[0],[1]],[[0],[numRows1],[1]],[[numCols1],[numRows1],[1]]]
    Hcorners = []
    for i in range(0,len(corners)):
        # multiply each corner by H matrix
        cor = matrix_multiply(opencvH,corners[i])
        scaledCor = []
        # divide by 3rd homography coordinate
        for j in range(0,len(cor)):
            scaledCor.append(cor[j][0] / cor[2][0])
        Hcorners.append(scaledCor)
    # Display corners 
    for i in Hcorners:
        print(i)

    #### find minX, minY, maxX, and maxY
    minX = int(min(Hcorners[0][0],Hcorners[1][0],Hcorners[2][0],Hcorners[3][0], 0, numCols2, 0, numCols2))
    maxX = int(max(Hcorners[0][0],Hcorners[1][0],Hcorners[2][0],Hcorners[3][0], 0, numCols2, 0, numCols2))
    minY = int(min(Hcorners[0][1],Hcorners[1][1],Hcorners[2][1],Hcorners[3][1], 0, 0, numRows2, numRows2))
    maxY = int(max(Hcorners[0][1],Hcorners[1][1],Hcorners[2][1],Hcorners[3][1], 0, 0, numRows2, numRows2))
    dimX = maxX - minX
    dimY = maxY - minY
    pano = np.zeros( (dimY,dimX,3), np.float32)
    shiftX = abs(minX)
    shiftY = abs(minY)

    ### Copy img2 over to panorama
    colorim1 = cv2.imread(im1Name,3)
    colorim2 = cv2.imread(im2Name,3)
    for i in range(numRows2):
        for j in range(numCols2):
            pano[shiftY + i][shiftX + j] = colorim2[i][j]

    Hinv = np.linalg.inv(opencvH)
    redDiff = 0
    blueDiff = 0
    greenDiff = 0
    pixel_sum = 0

    for i in range(dimY):
        for j in range(dimX):
            # Calculate mapped pixel
            pixel = [[j- shiftX],[i-shiftY],[1]]
            mapped = matrix_multiply(Hinv,pixel)
            scaled_mapped = []
            for k in range(0,len(mapped)):
                scaled_mapped.append(mapped[k][0]/mapped[2][0])

            # If the mapped pixel is in the bounds of image 1
            if scaled_mapped[0] > 0 and scaled_mapped[1]> 0 and scaled_mapped[0] < numCols1 and scaled_mapped[1] <numRows1:
                if np.all(pano[i][j] != [0, 0, 0]):
                    # calculate color differences
                    redDiff += colorim1[int(scaled_mapped[1])][int(scaled_mapped[0])][2] - pano[i][j][2] 
                    blueDiff += colorim1[int(scaled_mapped[1])][int(scaled_mapped[0])][0] - pano[i][j][0] 
                    greenDiff += colorim1[int(scaled_mapped[1])][int(scaled_mapped[0])][1] - pano[i][j][1]
                    pixel_sum += 1

                
    ### Stitch images together
    for i in range(dimY):
        for j in range(dimX):
            #Calculate mapped pixel
            pixel = [[j- shiftX],[i-shiftY],[1]]
            mapped = matrix_multiply(Hinv,pixel)
            scaled_mapped = []
            for k in range(0,len(mapped)):
                scaled_mapped.append(mapped[k][0]/mapped[2][0])

            # If mapped pixel is in the bounds of image 1
            if scaled_mapped[0] > 0 and scaled_mapped[1]> 0 and scaled_mapped[0] < numCols1 and scaled_mapped[1] <numRows1:
                # Copy each color channel over with calculated color difference
                pano[i][j][2] = colorim1[int(scaled_mapped[1])][int(scaled_mapped[0])][2] - (redDiff/pixel_sum)
                pano[i][j][0] = colorim1[int(scaled_mapped[1])][int(scaled_mapped[0])][0] - (blueDiff/pixel_sum)
                pano[i][j][1] = colorim1[int(scaled_mapped[1])][int(scaled_mapped[0])][1] - (greenDiff/pixel_sum)



    cv2.imshow("Stitched Images",pano/255)
    cv2.imshow("Im1",colorim1/255)
    cv2.imshow("Im2",colorim2/255)

    cv2.waitKey(0) #pause program, proceed when you hit "enter"
    cv2.destroyAllWindows() #closes all windows created with imhsow

    cv2.imwrite("Panorama.png" , pano)


if __name__ == "__main__":
    main()
