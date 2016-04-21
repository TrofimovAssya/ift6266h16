#%matplotlib inline
import skimage
import numpy
import scipy
import sys
import os
from skimage import io
from skimage import transform
import matplotlib.pyplot as plt

""" 
Script for image processing, according to Howard, A (http://arxiv.org/pdf/1312.5402.pdf) and Hinton et al. 
(https://www.nvidia.cn/content/tesla/pdf/machine-learning/imagenet-classification-with-deep-convolutional-nn.pdf)
Operations done:
1 - images are cropped to biggest squares in the middle
2 - images are resized to fit greyscaled 32 x 32 pixels 

Input images are taken from the path specified by the user at launch
Processed images are stored in the path specified by the user at launch
"""


### to the image files specified at script launch
path = sys.argv[1]
out_path = sys.argv[2]

# function that crops the image to a square of the smallest dimension
def crop(image,min_dim):
    length = numpy.shape(image)[1] 
    width = numpy.shape(image)[0]
    half = numpy.floor(min_dim/2)
    if not (length%2 == 0):
        length-=1
    if not (width%2 == 0):
        width-=1    
    top = numpy.floor(length/2.-half)
    bottom = numpy.ceil(length/2.+half)
    left = numpy.floor(width/2.-half)
    right = numpy.ceil(width/2.+half)
    return image[ left:right , top:bottom ]

# function that resizes the square image to the specified size
def resize(image,size):
	return scipy.misc.imresize(image,(size,size))

# function that goes throught the files and processes them
def process(path,size):
	for filename in os.listdir(path):
		if not(filename[-4:] == ".jpg"):
			continue

		image = io.imread("".join([path,str(filename)]), as_grey = True)
		min_dim = min(numpy.shape(image)[:2])
		if not(min_dim % 2 == 0):
			min_dim -= 1

		image = crop(image,min_dim)
		image = resize(image,size)
		print (str(filename[:-4]))
		io.imsave("".join([out_path,str(filename[:-4]),str("_processed.jpg")]) , image)

process(path,100)

