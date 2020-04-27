# Giovana Meloni Craveiro
# 9791264
# SCC0251_Turma01_1Sem_2020 - Image Processing 
# Assignment2 - Image Enhancement and Filtering
#

import numpy as np
import imageio

def Bilateral_Filter_1(img,n,sigmaS,sigmaR): # Method 1 - Bilateral Filter
	#function goes here


def Laplacian_Filter_2(img,c,kernel): # Method 2 - Unsharp Mask using the Laplacian Filter
	#function goes here

def Vignette_Filter_3(img,sigmaRow,sigmaCol): # Method 3 - Vignette Filter
	#function goes here


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

