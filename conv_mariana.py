import numpy
import cPickle as pickle

import Mariana.activations as MA
import Mariana.layers as ML
import Mariana.convolution as MCONV
import Mariana.costs as MC
import Mariana.regularizations as MR
import Mariana.scenari as MS
import Mariana.decorators as MD






####
# First attempt at a conv net based on the conv net example in Mariana examples
# original by Tariq Daouda
#

def load_data(picklefile):
#	tbl = numpy.load(open("small_set.npy"))
	tbl = pickle.load(open(picklefile,"rb"))

	return tbl

class ConvWithChanneler :
	
	def __init__(self, ls, cost) :
		maxPool = MCONV.MaxPooling2D(2, 2)
	
	#maxPool = MCONV.MaxPooling2D(2, 2)
		
	#The input channeler will take regular layers and arrange them into several channels
		i = MCONV.Input(nbChannels = 3, height = 256, width = 256, name = 'inp')
		#ichan = MCONV.InputChanneler(256, 256, name = 'inpChan')
		
		c1 = MCONV.Convolution2D( 
			nbFilters = 3,
			filterHeight = 5,
			filterWidth = 5,
			activation = MA.Tanh(),
			pooler = maxPool,
			name = "conv1"
		)

		c2 = MCONV.Convolution2D( 
			nbFilters = 3,
			filterHeight = 5,
			filterWidth = 5,
			activation = MA.Tanh(),
			pooler = maxPool,
			name = "conv2"
		)

		f = MCONV.Flatten(name = "flat")
		h = ML.Hidden(5, activation = MA.Tanh(), decorators = [MD.GlorotTanhInit()], regularizations = [MR.L1(0), MR.L2(0.0001)], name = "hid" )
		o = ML.SoftmaxClassifier(2, decorators = [], learningScenario = ls, costObject = cost, name = "out", regularizations = [ ] )
		
		self.model = i > c1 > c2 > f > h > o
	#return model
	
	def train(self, inputs, targets) :
	 	#because of the channeler there is no need to reshape the data besfore passing them to the conv layer
	 	return self.model.train("out", inp = inputs, targets = targets )

	def test(self,inputs,targets) :
		return self.model.test("out", inp = inputs, targets = targets )

	def saveHTML(self, name) :
		return self.model.saveHTML(name)

if __name__ == "__main__" :
	
	ls = MS.GradientDescent(lr = 1e-3)
	cost = MC.NegativeLogLikelihood()

	maxEpochs = 200
	miniBatchSize = 10
	
	model = ConvWithChanneler(ls, cost)
	model.saveHTML('cnn_catdog')
	datafile = "small_set"
	train_set = load_data(datafile)


	#shuffling the dataset
	indices = [i for i in xrange(len(train_set[0]))]
	numpy.random.shuffle(indices)

	train_set = (train_set[0][indices],train_set[1][indices])

	vdata = (train_set[0][:int(0.2*len(train_set[0]))],train_set[1][:int(0.2*len(train_set[0]))])
	tdata = (train_set[0][int(0.2*len(train_set[0])):],train_set[1][ int(0.2*len(train_set[0])):])
	tscore = []
	vscore = []

	epoch = 0
	while True :
		trainScores = []
		for i in xrange(0, len(tdata), miniBatchSize) :
			inputs = tdata[0][i : i +miniBatchSize]
			targets = tdata[1][i : i +miniBatchSize]
			res = model.train(inputs, targets)
#			a = model.train('out', inputs = data[i:i+miniBatchSize], targets = targets[i:i+miniBatchSize])
			trainScores.append(res[0])
	
		trainScore = numpy.mean(trainScores)
		
		print "---\nepoch", epoch
		print "\ttrain score:", trainScore
		tscore.append(trainScore)

		validScore = model.test(vdata[0],vdata[1])[0]

		print "\tvalid score:", validScore.item()
		vscore.append(validScore.item())
		if epoch % 10 == 0:
			numpy.save('tscores',tscore)
			numpy.save('vscores',vscore)
		epoch += 1
