# Giovana Meloni Craveiro
# 9791264
# SCC0251_Turma01_1Sem_2020 - Image Processing 
# Assignment2 - Image Enhancement and Filtering
#

import numpy as np
import imageio
#import matplotlib.pyplot as plt

def Bilateral_Filter_1(img,n,sigmaS,sigmaR): # Method 1 - Bilateral Filter
    
    gr_f = np.zeros((n,n),dtype=np.float) # creates matrix to store range Gaussian
    gs_f = np.zeros((n,n),dtype=np.float) # creates matrix to store spatial Gaussian component

    a = int((n-1)/2) # (a,a) is the value of the center of the filter, a can be used to adequately walk through the neighborhood of each pixel, with its center placed at the origin, according to n given

    # CREATING FILTER

    #Gaussian Spatial Component
    for x in range(-a,a+1): # values of x and y of each position of an n-sized matrix are needed to calculate euclidean distance from each position to the center
        for y in range(-a,a+1):
            euc = np.sqrt((np.square(x) + np.square(y))) #applies the euclidean distance between the position and the center to find the x value to use in Gaussian kernel equation
            gs_i = float(1) / (2*np.pi*np.square(sigmaS)) #applies the Gaussian kernel equation G(euc,sigmaS) for each pixel of the gaussian spatial component
            gs_i = gs_i * (np.exp( float(-np.square(euc)) / float(( 2*np.square(sigmaS)) ) ) )
            gs_f[x,y] = float(gs_i)

    height, width = img.shape # gets height and width of the original image
    subimg = np.zeros((n,n),dtype=np.float) #creates submatrix to store the current neighborhood of the point being calculated
    result_img = np.zeros(img.shape,dtype=np.uint8) #creates a new empty image with size nxn to store the final image

    #Convolution
    for i in range (a,height-a): # walks through the final image
        for j in range(a,width-a):

            I_f = float(0) # value for the pixel to be calculated is initiated at 0
            W_p = float(0) # normalization factor for the pixel is initiated at 0

            subimg = img[(i-a):(i+a+1), (j-a):(j+a+1)] #gets current submatrix of pixel neighborhood in order to apply filter on it

            for x in range(n): # walks through neighborhood matrix of the current point
                for y in range(n):
                    #Range Gaussian Component
                    dif = float(subimg[x,y]) - float(subimg[a,a])# calculates distance between the current position and the center of the neighborhood matrix, in order to find the x value to use in Gaussian kernel equation
                    gr_i = 1 / (2*np.pi*np.square(sigmaR)) # applies the Gaussian Kernel equation G(dif,sigmaR) for each pixel of the range gaussian component
                    gr_i = gr_i * (np.exp( -np.square(dif) / ( 2*np.square(sigmaR) ) ) )
                    gr_f[x,y] = float(gr_i)

                    w_i = float(gs_f[x,y])*float(gr_f[x,y]) # calculates the value of the filter for each pixel in the neighborhood by multiplying tha range Gaussian and the spatial Gaussian components

                    W_p = W_p + float(w_i) # calculates normalization factor 

                    I_f = I_f + float(w_i*subimg[x,y]) # applies the filter to each point of the neighborhood, by multiplying the filter local value and the neighborhood pixel and summing them all
     
            result_img[i,j] = int(I_f/W_p) # fills up position [x,y] of the final image with the correct number    
    
    return result_img

def Laplacian_Filter_2(img,c,kernel): # Method 2 - Unsharp Mask using the Laplacian Filter
    
    k = np.zeros((3,3),dtype=np.float)

    if kernel == 1:
        k = [[0,-1,0],[-1,4,-1],[0,-1,0]] # creates kernel 1
    elif kernel == 2:
        k = [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]] #creates kernel 2

    height, width = img.shape
    neighborhood = np.zeros((3,3),dtype=np.float) #creates submatrix to store the current neighborhood of the point being calculated
    result_img = np.zeros(img.shape,dtype=np.uint8) #creates a new empty image with size nxn to store the final image

    #Convolution with chosen kernel
    for i in range(1,height-1):
        for j in range(1,width-1):

            neighborhood = img[(i-1):(i+2),(j-1):(j+2)]
            img_pixel = float(0)

            for x in range(3):
                for y in range(3):
                    neigh_pix = float(k[x,y])*float(neighborhood[x,y])
                    img_pixel = img_pixel + neigh_pix
                    
            result_img[i,j] = img_pixel
    
    result_img = (result_img - np.min(result_img))*255/np.max(result_img) # scale the filtered image using normalization (0 - 255)

    result_img = c*result_img + img # adds the filtered image, multiplied by c, back to the original image

    result_img = (result_img - np.min(result_img))*255/np.max(result_img) # scale the fial image using normalization (0-255)

    return result_img

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

    a = int((n-1)/2) # calculates size of the padding required
    image = np.pad(image,(a,a),mode='constant',constant_values=(0)) #adds image padding

    result_img = Bilateral_Filter_1(image,n,sigmaS,sigmaR)

if M == 2: # Unsharp Mask using the Laplacian Filter

    c = float(input())

    kernel = int(input())

    image = np.pad(image,(1,1),mode='constant',constant_values=(0)) #adds image padding
    
    result_img = Laplacian_Filter_2(image,c,kernel)


if M == 3: # Vignette Filter

    sigmaRow = float(input())

    sigmaCol = float(input())

    result_img = Vignette_Filter_3(image,sigmaRow,sigmaCol)

# 3 - Compare the new image with the reference image using the following formula: RSE = sqrt(∑i∑j (f(i,j) - r(i,j))²), in which f is the modified image and r is the reference image

rse = (np.sqrt( np.sum ( np.square (np.subtract(result_img.astype(np.int32),image.astype(np.int32)) ) ) ) )

# 4 - Print in the screen the root squared error between the images, which I named rse

print("%.4f" % rse)

# 5 - Save the filtered image, if S = 1

if S == 1:
    imageio.imwrite("FilteredImage.jpg",result_img)

