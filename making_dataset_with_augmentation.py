# processing dataset files to fit the convnet
# the input are the files processed by image_processing.py
# the output is one large pickle file containing the training data with all pictures and their classifications
# as well as the pickle file containing the validation data with pictures and classifications.

import sys
import os
import numpy
import cPickle as pickle
from PIL import Image
from numpy import random



#generating custom validation set as specified by teacher
indices = numpy.arange(25000)
numpy.random.RandomState(123522).shuffle(indices)
valid = indices[-2500:]


#obtaining the file lists
filelist = os.listdir("./")

# function that processes the training sets
def process_train(filelist):
	data = []
	targets = []
	for i in xrange (25000):
		filename = filelist[i]
		names = filename.split(".")
		if not names[-1] == "jpg":
			continue

		if i in valid:
			continue

		print filename

		img = Image.open(open("/".join([".",filename])))
		img = numpy.array(img, dtype='float32') / 256.
		## swap axes is a hack for the picky Mariana Convnet input.
#		img = numpy.swapaxes(img,0,2)
		if (names[0] == 'cat'):
			animal = 1
		else:
			animal = 0

		data.append(img)
		targets.append(animal)

		## the data augmentation means you add twice more images with horizontal flips!
		data.append(numpy.fliplr(img))
		targets.append(animal)

	return data,targets


#making a validation set
def process_valid(vlist, filelist):
	data = []
	targets = []
	for i in vlist:
		filename = filelist[i]
		names = filename.split(".")
		print filename
		img = Image.open(open("/".join([".",filename])))
		img = numpy.array(img, dtype='float32') / 256.
#		img = numpy.swapaxes(img,0,2)
		if (names[0] == 'cat'):
			animal = 1
		else:
			animal = 0


		data.append(img)
		targets.append(animal)

	return data,targets


def process_test(filelist):
	data = []
	for i in xrange (12500):
		filename = filelist[i]
		names = filename.split(".")
		if not names[-1] == "jpg":
			continue
		print filename
		img = Image.open(open("/".join(["./",filename])))
		img = numpy.array(img, dtype='float32') / 256.
	
		data.append(img)

	return data

# processing the training sets 

data,targets = process_train(filelist)
data = numpy.array(data)
targets = numpy.array(targets)
train_set = (data,targets)
pickle.dump(train_set,open("train_dataset.p",'wb'))

#making a validation set
data,targets = process_valid(valid,filelist)
data = numpy.array(data)
targets = numpy.array(targets)
valid_set = (data,targets)
pickle.dump(valid_set,open("valid_dataset.p",'wb'))

# #making the test set
# data = process_test(filelist)
# data = numpy.array(data)
# pickle.dump(data,open("test_set.p",'wb'))
