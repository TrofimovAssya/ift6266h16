import numpy
import cPickle as pickle

import Mariana.activations as MA
import Mariana.layers as ML
import Mariana.convolution as MCONV
import Mariana.costs as MC
import Mariana.regularizations as MR
import Mariana.scenari as MS
import Mariana.decorators as MD

from convnet import ConvWithChanneler

####
# using previous version of the Cat-Dog identification convnet
# solution 1 for dealing with a large dataset:
# load the pictures by batches of pre-processed pickle files
# shuffle after each load and split validation data into an array.

def load_data_batch(picklefile):
	tbl = pickle.load(open(picklefile,"rb"))
	return tbl



if __name__ == "__main__" :
	
	ls = MS.GradientDescent(lr = 1e-3)
	cost = MC.NegativeLogLikelihood()

	miniBatchSize = 10
	
	model = ConvWithChanneler(ls, cost)

	#printing the model to view the architecture
	model.saveHTML('cnn_catdog')
	
	tscore = []
	vscore = []

	epoch = 0
	while epoch<1000 :
		trainScores = []
		for batchindex in xrange(1,126,1):
			print "loading batch", batchindex
			tdata = load_data_batch(("_").join([str(batchindex),"dataset.p"]))
#			for i in xrange(0, len(tdata), miniBatchSize) :
			inputs = tdata[0]
			targets = tdata[1]
			res = model.train(inputs, targets)
			trainScores.append(res[0])
#				if i%10 == 0:
			print res[0]
		
		trainScore = numpy.mean(trainScores)
		
		print "---\nepoch", epoch
		print "\ttrain score:", trainScore
		tscore.append(trainScore)

		vdata = load_data_batch("valid_dataset.p")
		validScore = model.test(vdata[0],vdata[1])[0]

		print "\tvalid score:", validScore.item()
		vscore.append(validScore.item())
		if epoch % 10 == 0:
			numpy.save('tscores',tscore)
			numpy.save('vscores',vscore)
		epoch += 1
