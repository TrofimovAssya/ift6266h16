import numpy
import cPickle as pickle

import Mariana.activations as MA
import Mariana.layers as ML
import Mariana.convolution as MCONV
import Mariana.costs as MC
import Mariana.regularizations as MR
import Mariana.scenari as MS
import Mariana.decorators as MD


class ConvWithChanneler :
	
	def __init__(self, ls, cost) :
		maxPool = MCONV.MaxPooling2D(2,2)
		i = MCONV.Input(nbChannels = 1, height = 100, width = 100, name = 'inp')

		ichan = MCONV.InputChanneler(100, 100, name = 'inpChan')
		
		c1 = MCONV.Convolution2D( 
			nbFilters = 1,
			filterHeight = 20,
			filterWidth = 20,
			activation = MA.ReLU(),
			pooler = maxPool,
			name = "conv1"
		)

		c2 = MCONV.Convolution2D( 
			nbFilters = 1,
			filterHeight = 5,
			filterWidth = 5,
			activation = MA.ReLU(),
			pooler = maxPool,
			name = "conv2"
		)

		f = MCONV.Flatten(name = "flat")
		h = ML.Hidden(1000, activation = MA.ReLU(), decorators = [MD.BinomialDropout(0.2)], regularizations = [ ], name = "hid" )
		o = ML.SoftmaxClassifier(2, decorators = [], learningScenario = ls, costObject = cost, name = "out", regularizations = [MR.L1(1e-7) ] )
		
		self.model = i > c1 > c2 > f > h > o

	#The input channeler will take regular layers and arrange them into several channels
# 		i = MCONV.Input(nbChannels = 3, height = 32, width = 32, name = 'inp')
# 		#ichan = MCONV.InputChanneler(256, 256, name = 'inpChan')

# 		c3 = MCONV.Convolution2D( 
# 			nbFilters = 1,
# 			filterHeight = 20,
# 			filterWidth = 20,
# 			activation = MA.ReLU(),
# 			pooler = MCONV.NoPooling(),
# 			name = "conv3"
# 		)

# 		c4 = MCONV.Convolution2D( 
# 			nbFilters = 1,
# 			filterHeight = 13,
# 			filterWidth = 13,
# 			activation = MA.ReLU(),
# 			pooler = MCONV.MaxPooling2D(2,2),
# 			name = "conv4"
# 		)

# 		f = MCONV.Flatten(name = "flat")
# #		h = ML.Hidden(3, activation = MA.ReLU(), decorators = [MD.BinomialDropout(0.01)], regularizations = [], name = "hid" )
# 		o = ML.SoftmaxClassifier(2, decorators = [], learningScenario = ls, costObject = cost, name = "out", regularizations = [MR.L1(1e-4) ] )
		
# 		self.model = i > c3 > c4 > f > o

	
	def train(self, inputs, targets) :
	 	#because of the channeler there is no need to reshape the data besfore passing them to the conv layer
	 	inputs = inputs.reshape((-1, 1, 100, 100))
	 	return self.model.train("out", inp = inputs, targets = targets )

	def test(self,inputs,targets) :
	 	inputs = inputs.reshape((-1, 1, 100, 100))
		return self.model.test("out", inp = inputs, targets = targets )

	def propagate(self,inputs) :
	 	inputs = inputs.reshape((-1, 1, 100, 100))
		return self.model.propagate("out", inp = inputs)

	def saveHTML(self, name) :
		return self.model.saveHTML(name)
