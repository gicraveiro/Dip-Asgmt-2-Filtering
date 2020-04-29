# Giovana Meloni Craveiro
# 9791264
# SCC0251_Turma01_1Sem_2020 - Image Processing 
# Assignment2 - Image Enhancement and Filtering
#

import numpy as np
import imageio
import matplotlib.pyplot as plt

def Bilateral_Filter_1(img,n,sigmaS,sigmaR): # Method 1 - Bilateral Filter
    #function goes here

    w = np.zeros((n,n),dtype=np.float)#.astype(np.int32) #creates matrix of size nxn
    #n,n = filter.shape

    a = int((n-1)/2) # a = b since the filter has size nxn
    #create filter

    for x in range(-a,a+1): #assuming the filter in the center to use correct values for x and y
        for y in range(-a,a+1):
            euc = float(np.sqrt(np.square(x) + np.square(y)) )#apply the euclidean distance between the position and the center to find the x value
            gauss_point = float( np.exp( -np.square(euc) / ( 2*np.square(sigmaS) ) ) / (2*np.pi*np.square(sigmaS))) #apply the Gaussian Kernel equation G(x,y) for each pixel
            w[x,y] = gauss_point#.astype(np.int32)
            print(w[x,y])

    print("Let's see how the filter is being created...\n")
    print(w) #expect filter to be created correctly

   # return img;

    #get submatrix of pixel neighborhood to apply filter on the pixel
    #filter = np.matrix()
    subimg = np.zeros((n,n),dtype=np.float)
    subimg = img[(x-a):(x+a+1), (y-a):(y+a+1)]
    print(subimg)
    print(img)
    #flips the filter
    #subimg = np.flip (np.flip(subimg,0), 1)
    print(subimg)

    result_img = np.empty(w.shape,dtype=np.uint8) #creates a new empty image with size nxn
    print(result_img)

    #loops through each pixel of the image calculating the new value, through the convolution point
    for x in range(n):
        for y in range(n):
            conv_point = np.sum(np.multiply(subimg, w)) # sum of each of the neighborhood pixel multiplied by the filter is the convolution point
            result_img[x,y] = conv_point#.astype(np.uint8) 

    print(result_img)
    
    return result_img

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

    image = np.pad(image,(1,1),mode='constant',constant_values=(0))
    print(image)
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

#rse = np.sqrt( np.sum ( np.square( np.subtract(result_img.astype(np.int32),image.astype(np.int32)) ) ))
rse = (np.sqrt( np.sum ( np.square (np.subtract(result_img.astype(np.int32),image.astype(np.int32)) ) ) ) )

# 4 - Print in the screen the root squared error between the images, which I named rse

print("%.4f" % rse)

# 5 - Save the filtered image, if S = 1

if S == 1:
    imageio.imwrite("FilteredImage.jpg",result_img)

# showing images
plt.figure(figsize=(12,12)) 
plt.subplot(121)
plt.imshow(image, cmap="gray", vmin=0, vmax=255)
plt.title("original, noisy")
plt.axis('off')
plt.subplot(122)
plt.imshow(result_img, cmap="gray", vmin=0, vmax=255)
plt.title("filtered image")
plt.axis('off')