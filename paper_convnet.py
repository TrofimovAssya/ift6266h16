import numpy
from time import clock
from random import shuffle
import cPickle as pickle
import sys
import Mariana.activations as MA
import Mariana.layers as ML
import Mariana.convolution as MCONV
import Mariana.costs as MC
import Mariana.regularizations as MR
import Mariana.scenari as MS
import Mariana.decorators as MD

from papernet import ConvNet

####
# using a simplified version of the Convnet described by Srivastava et. al. 
# https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
#
# Each epoch the train data is shuffled
# The data is centered (see center() function)

def load_data(picklefile):
	tbl = pickle.load(open(picklefile,"rb"))
	return tbl

def center(batch):
	mean = numpy.mean(batch)
	sd = numpy.std(batch)
	return (batch-mean)/sd

def checkout(predict,target):
        a = zip(predict,target)
        correct = 0
        for i in a:
                if numpy.argmax(i[0]) == i[1]:
                        correct+=1
        return correct/float(len(target))

def classes(targets):
        x = [numpy.abs(numpy.argmax(i)-1) for i in targets[0]]
        return x

if __name__ == "__main__" :
	
	ls = MS.MomentumGradientDescent(lr = 1e-1, momentum = 0.95)
	cost = MC.NegativeLogLikelihood()

	miniBatchSize = 100
	
	model = ConvNet(ls, cost)

	#printing the model to view the architecture
#	model.saveHTML('cnn_catdog')
	
	tscore = []
	vscore = []
	tdata = load_data("../flip_grey_train_dataset.p")
  	vdata = load_data("../flip_grey_valid_dataset.p")
    vdata = (center(vdata[0]),vdata[1])        
    test = load_data("../test_set.p")
    test = center(test)

	epoch = 0
	miniBatchSize = 100
	permutate = [i for i in xrange(len(tdata[0]))]
    permut2 = [i for i in xrange(len(vdata[0]))]

    #number of epochs is arbitrary, since I am using a checkpoint save
	while epoch<4000 :
		trainScores = []
		numpy.random.shuffle(permutate)
		tdata = (tdata[0][permutate],tdata[1][permutate])

		for i in xrange(0, len(tdata[0]), miniBatchSize) :
			inputs = center(tdata[0][i : i +miniBatchSize])
			targets = tdata[1][i : i +miniBatchSize]
			res = model.train(inputs, targets )
			trainScores.append(res[0])
			#making sure the python buffer is flushed
			sys.stdout.flush()
		trainScore = numpy.mean(trainScores)

		tscore.append(trainScore)
                vdata = (vdata[0][permut2], vdata[1][permut2])
		validScore = model.test(vdata[0],vdata[1])[0]
		print "epoch", epoch
		print "\ttrain score:", trainScore
		print "\tvalid score:", validScore.item()

		vscore.append(validScore.item())
		if epoch % 10 == 0:
			numpy.save('HD1Ctscores_grey',tscore)
			numpy.save('HD1Cvscores_grey',vscore)
			numpy.save('HD1Cerrors_grey',erate)
                        ccc = model.propagate(test)
                        ccc = classes(ccc)
                        numpy.save("_".join([str(epoch),"HD1Ctest_scores"]),ccc)

		epoch += 1


