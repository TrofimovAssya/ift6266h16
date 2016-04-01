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
		maxPool = MCONV.MaxPooling2D(2, 2)
	
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