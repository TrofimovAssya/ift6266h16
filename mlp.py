import theano
import numpy
import gzip
import cPickle as pickle 
from theano import tensor as T

################# this MLP is build as an example to classify patients by gender ################                                                                                  
########## architecture:                                                                                                                                                           
###   28*28 input                                                                                                                                                                   
### 200 hidden                                                                                                                                                                      
### 10 output                                                                                                                                                                       
###############data set building ####################                                                                                                                              
#############                                                                                                                                                                      
print "hello! I am learning to classify patients by gender using gene expression!"


#### making a numpy array with theano config, used by the initial_weights function                                                                                                 
def theanoFloat(X):
    return numpy.asarray(X,dtype = theano.config.floatX)

##### intializing the weight matrix                                                                                                                                                
def initial_weights (len_wid):
    return theano.shared(theanoFloat(numpy.random.randn(*len_wid) * 0.01))

##### stochastic gradient descent function, used to update model                                                                                               
##### the function returns a table containing the parameters - gradient * learningrate for each parameter                                                                                    
def sgd (cost, params, learningrate = 0.001):
    gradient = T.grad (cost = cost, wrt = params)
    updates = []
    for parameters, gradients in zip (params, gradient):
        updates.append([parameters, parameters - gradients * learningrate ])
    return updates

##### model is a function that shows the net architecture                                                                                                                          
##### the function calculates h, the output of the hidden layer and then pyx                                                                                                       
##### the output of the output layer, which is also the prediction of y given x                                                                                                    
def model (X, w1, w2):
    hidden = T.nnet.sigmoid(T.dot(X,w1)) # hidden layer activation                                                                                                                     
    predict = T.nnet.softmax(T.dot(hidden,w2)) # output softmax                                                                                                                            
    return predict
### theano  matrices x and y                                                                                                                                                       
X = T.fmatrix()
Y = T.lvector()

### weights matrices for the hidden and output layers                                                                                                                              
w1 = initial_weights ((28*28,200))
w2 = initial_weights ((200,10))

#### prediction of y given x                                                                                                                                                       
predict = model(X, w1, w2)
result = T.argmax(predict, axis=1)


#### cost and parameters                                                                                                                                                           
cost = T.mean(T.nnet.categorical_crossentropy(predict, Y))
params = [w1, w2]


#### train function takes in inputs, outputs cost and updates the model using the sgd function                                                                                     
#### which uses cost and params                                                                                                                                                    
train = theano.function(
            inputs = [X,Y],
            outputs = cost,
            updates = sgd(cost,params),
            allow_input_downcast = True)

#### function that calculates the cost but DOES NOT TRAIN THE MODEL                                                                                                                
valid = theano.function(
            inputs = [X,Y],
            outputs = T.mean(T.nnet.categorical_crossentropy(model(X, w1, w2), Y)),
            allow_input_downcast = True)


test = theano.function(
            inputs = [X],
            outputs = result,
            allow_input_downcast = True)

### data will be stored in results                                                                                                                                                 
##### training here                                                                                                                                                                
### validation is done after each epoch                                                                                                                                            
results = []

print "loading data...."
with gzip.open('mnist.pkl.gz', 'rb') as f:
    ts,vs,tes = pickle.load(f)


print "training...."
for j in range(1000):

    for i in range(0,len(ts), 1):
        cost = train([ts[i][0]], [ts[i][1]])
    for k in range(0,len(vaX), 1):
       cost2 = valid([vaX[k,]], [vaY[k,]])

    results.append([j, cost,cost2])

    print (('epoch %d training error: %f')%(j, cost))
    numpy.savetxt('results_1000.csv',results,delimiter=',')

#########testing the model                                                                                                                                                         
print 'testing the model'

model_test = []
count = 0
for i in range(0,len(teX),1):
    model_p = test([teX[i,]])
    print (('patient %d : predicted digit -> %f, actual -> %f')% (i,model_p,teY[i,]))
    model_test.append([i,model_p,teY[i,]])
    if model_p == teY[i,]:
        count = count+1

pct = float(count)/len(teY)*100
print pct
numpy.savetxt('testing_model_1000.csv',model_test,delimiter='\t')
