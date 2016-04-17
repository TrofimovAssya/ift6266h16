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

from grey_convnet import ConvWithChanneler

####
# using previous version of the Cat-Dog identification convnet
# solution 1 for dealing with a large dataset:
# load the pictures by batches of pre-processed pickle files
# shuffle after each load and split validation data into an array.

def load_data(picklefile):
	tbl = pickle.load(open(picklefile,"rb"))
	return tbl

def batch_norm(batch):
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

if __name__ == "__main__" :
	
	ls = MS.MomentumGradientDescent(lr = 1e-3, momentum = 0.1)
	cost = MC.NegativeLogLikelihood()

	miniBatchSize = 100
	
	model = ConvWithChanneler(ls, cost)

	#printing the model to view the architecture
#	model.saveHTML('cnn_catdog')
	
	tscore = []
	vscore = []
	erate = []
	tdata = load_data("flip_grey_train_dataset.p")
    vdata = load_data("flip_grey_valid_dataset.p")
    test = pickle.load(open("test_set.p","rb"))
	epoch = 0
	miniBatchSize = 100
	f = 0
	permutate = [i for i in xrange(len(tdata[0]))]
        permut2 = [i for i in xrange(len(vdata[0]))]
	while epoch<2000 :
		trainScores = []
		numpy.random.shuffle(permutate)
		tdata = (tdata[0][permutate],tdata[1][permutate])
#		t0 = clock()
		for i in xrange(0, len(tdata[0]), miniBatchSize) :
			inputs = batch_norm(tdata[0][i : i +miniBatchSize])
			targets = tdata[1][i : i +miniBatchSize]
			res = model.train(inputs, targets )
			trainScores.append(res[0])
#			if i%10000 == 0:
#				print "mini batch ", i, " training score:", res[0]
			sys.stdout.flush()
		trainScore = numpy.mean(trainScores)

		tscore.append(trainScore)
                vdata = (vdata[0][permut2], vdata[1][permut2])
		validScore = model.test(batch_norm(vdata[0]),vdata[1])[0]
		classes = model.propagate(vdata[0])
#		print classes
#		print vdata[1]
		print "epoch", epoch
		#if epoch==0:
		#	f = trainScore
		print "\ttrain score:", trainScore
		#	v = validScore
		print "\tvalid score:", validScore.item()
		#else:
		#	print "\ttrain score:", f-trainScore
		#	print "\ttrain score:", v-validScore.item()


		vscore.append(validScore.item())
		if epoch % 10 == 0:
			numpy.save('Dtscores_grey',tscore)
			numpy.save('Dvscores_grey',vscore)
                        erate.append(checkout(classes[0], vdata[1]))
			numpy.save('Derrors_grey',erate)
                        if epoch % 100 == 0: 
                                model.save("_".join([str(epoch),"Dgrey_model.p"]))
                                numpy.savetxt("_".join[str(epoch),"Dtest_scores"],model.propagate2(test))

		epoch += 1
#		print clock()-t0

