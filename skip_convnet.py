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


def load_data(picklefile):
	tbl = pickle.load(open(picklefile,"rb"))
	return tbl

def center(batch):
	mean = numpy.mean(batch)
	sd = numpy.std(batch)
	return (batch-mean)/sd

def classes(targets):
        x = [numpy.abs(numpy.argmax(i)-1) for i in targets[0]]
        return x

maxPool = MCONV.MaxPooling2D(2,2)
ls = MS.MomentumGradientDescent(lr = 1e-1, momentum = 0.95) 
cost = MC.NegativeLogLikelihood()
miniBatchSize = 100
trainfile = "../flip_grey_train_dataset.p"
validfile = "../flip_grey_valid_dataset.p"
testfile = "../test_set.p"
runprefix = "HD2"


i = MCONV.Input(nbChannels = 1, height = 100, width = 100, name = 'inp')
		
c1 = MCONV.Convolution2D( 
	nbFilters = 15,
	filterHeight = 3,
	filterWidth = 3,
	activation = MA.Max_norm(),
	pooler = maxPool,
	name = "conv1"
)

c3 = MCONV.Convolution2D( 
	nbFilters = 25,
	filterHeight = 3,
	filterWidth = 3,
	activation = MA.Max_norm(),
	pooler = maxPool,
	name = "conv3"
)

c2 = MCONV.Convolution2D( 
	nbFilters = 15,
	filterHeight = 3,
	filterWidth = 3,
	activation = MA.Max_norm(),
	pooler = maxPool,
	name = "conv2"
)
fa = MCONV.Flatten(name="flata")
fb = MCONV.Flatten(name="flatb")
f = MCONV.Flatten(name = "flat")

h = ML.Hidden(2048, activation = MA.Max_norm(), decorators = [MD.BinomialDropout(0.75)], regularizations = [], name = "hid" )
passa = ML.Hidden(1500, activation = MA.Pass(), decorators = [MD.BinomialDropout(0.5)], regularizations = [], name = "pass1" )
passb = ML.Hidden(1500, activation = MA.Pass(), decorators = [MD.BinomialDropout(0.5)], regularizations = [], name = "pass2" )
h2 = ML.Hidden(2048, activation = MA.Max_norm(), decorators = [MD.BinomialDropout(0.75)], regularizations = [], name = "hid2" )
o = ML.SoftmaxClassifier(2, decorators = [], learningScenario = ls, costObject = cost, name = "out", regularizations = [] )

model = i > c1 > c3 > c2 > f > h > h2 > o
c1 > fa > passa > h > h2 > o
c2 > fb > passb >h > h2 > o

tscore = []
vscore = []
tdata = load_data(trainfile)
vdata = load_data(validfile)
vdata = (center(vdata[0]),vdata[1])        
test = load_data(testfile)
test = center(test)

epoch = 0
permutate = [i for i in xrange(len(tdata[0]))]
permut2 = [i for i in xrange(len(vdata[0]))]

#number of epochs is arbitrary, since I am using a checkpoint save
while epoch<4000 :
	trainScores = []
	numpy.random.shuffle(permutate)
	tdata = (tdata[0][permutate],tdata[1][permutate])
	call = []
	for i in xrange(0, len(tdata[0]), miniBatchSize) :
		inputs = center(tdata[0][i : i +miniBatchSize])
		targets = tdata[1][i : i +miniBatchSize]
		inputs = inputs.reshape((-1, 1, 100, 100))
		res = model.train("out", inputs, targets )
		trainScores.append(res[0])

		#making sure the python buffer is flushed
		sys.stdout.flush()
	trainScore = numpy.mean(trainScores)

	tscore.append(trainScore)
    vdata = (vdata[0][permut2], vdata[1][permut2])
	validScore = model.test(vdata[0].reshape((-1, 1, 100, 100)),vdata[1])[0]
	print "epoch", epoch
	print "\ttrain score:", trainScore
	print "\tvalid score:", validScore.item()

	vscore.append(validScore.item())
	if epoch % 10 == 0:
		numpy.save("_".join([str(runprefix),"test_scores"]),tscore)
		model.save("_".join([str(runprefix),"model"]))
		numpy.save("_".join([str(runprefix),"test_scores"]),vscore)
        ccc = model.propagate(test.reshape((-1, 1, 100, 100)))
        ccc = classes(ccc)
        numpy.save("_".join([str(epoch),str(runprefix),"test_scores"]),ccc)

	epoch += 1



