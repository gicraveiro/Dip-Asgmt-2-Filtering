# Giovana Meloni Craveiro
# 9791264
# SCC0251_Turma01_1Sem_2020 - Image Processing 
# Assignment2 - Image Enhancement and Filtering
#

import numpy as np
import imageio

def Bilateral_Filter_1(img,n,sigmaS,sigmaR): # Method 1 - Bilateral Filter
    #function goes here

    n,n = filter.shape

    #create filter

    #apply the euclidean distance between the position and the center to find the x value
    
    x = np.sqrt(np.square(x) + np.square(y))

    #apply the Gaussian Kernel equation G(x,y) for each pixel

    gauss_point = np.exp( -np.square(x) / ( 2*np.square(sigmaS) ) ) / (2*np.pi*np.square(sigmaS))
    

    a = int((n-1)/2) # a = b since the filter has size nxn

    #get submatrix of pixel neighborhood to apply filter on the pixel
    #filter = np.matrix()
    subimg = f[x-1:x+2, y-1:y+2]

    #flips the filter
    subimg = np.flip (np.flip(subimg,0), 1)

    result_img = np.empty(filter.shape,stype=np.uint8) #creates a new empty image with size nxn

    #loops through each pixel of the image calculating the new value, through the convolution point
    for x in range(n):
        for y in range(n):
            conv_point = np.sum(np.multiply(subimg, filter)) # sum of each of the neighborhood pixeld multiplied by the filter is the convolution point
            result_img[x,y] = conv_point.astype(np.uint8) 

def Laplacian_Filter_2(img,c,kernel): # Method 2 - Unsharp Mask using the Laplacian Filter
    #function goes here
    aux = 1 #delete later

def Vignette_Filter_3(img,sigmaRow,sigmaCol): # Method 3 - Vignette Filter
    #function goes here
    aux = 1 #delete later

imagename = str(input()).rstrip() # reads the name of the reference image file

image = imageio.imread(imagename) # reads the image

M = int(input()) # paramater to indicate the method 1,2, our 3

S = int(input()) # parameter to know if the image should be saved, 1 for yes


if M == 1: # Bilateral Filter

    n = int(input()) #size of the filter n 

    sigmaS = float(input()) 

    sigmaR = float(input())

    result_img = Bilateral_Filter_1(image,n,sigmaS,sigmaR)

if M == 2: # Unsharp Mask using the Laplacian Filter

    c = float(input())

    kernel = int(input())
    
    result_img = Laplacian_Filter_2(image,c,kernel)

if M == 3: # Vignette Filter

    sigmaRow = float(input())

    sigmaCol = float(input())

    result_img = Vignette_Filter_3(image,sigmaRow,sigmaCol)

# 3 - Compare the new image with the reference image using the following formula: RSE = sqrt(∑i∑j (f(i,j) - r(i,j))²), in which f is the modified image and r is the reference image

rse = np.sqrt(np.sum( np.sum( np.square( np.subtract(result_img.astype(np.int32),image.astype(np.int32)) ) ) ))

# 4 - Print in the screen the root squared error between the images, which I named rse

print("%.4f" % rse)

# 5 - Save the filtered image, if S = 1

if S == 1:
    imageio.imwrite("FilteredImage.jpg",result_img)

