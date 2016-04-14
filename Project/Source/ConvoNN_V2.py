import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.signal import downsample
import numpy
import numpy as np
import time
import pylab
from PIL import Image
from Data import Load
import csv
import os
import pickle
import sys


from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

## (1) Parameters
numClasses = 30
numHiddenNodes = 600 
patchWidth = 3
patchHeight = 3
featureMapsLayer1 = 32
featureMapsLayer2 = 64
featureMapsLayer3 = 128

# For convonets, we will work in 2d rather than 1d.  The dataset images are 96 * 96
imageWidth = 96

# Convolution layers.  
w_1 = theano.shared(floatX(np.asarray((np.random.randn(*(featureMapsLayer1, 1, patchWidth, patchHeight))*.01))))
w_2 = theano.shared(floatX(np.asarray((np.random.randn(*(featureMapsLayer2, featureMapsLayer1, patchWidth, patchHeight))*.01))))
w_3 = theano.shared(floatX(np.asarray((np.random.randn(*(featureMapsLayer3, featureMapsLayer2, patchWidth, patchHeight))*.01))))

# Fully connected NN. 
w_4 = theano.shared(floatX(np.asarray((np.random.randn(*(featureMapsLayer3 * 11 * 11, numHiddenNodes))*.01))))
w_5 = theano.shared(floatX(np.asarray((np.random.randn(*(numHiddenNodes, numHiddenNodes))*.01))))
w_6 = theano.shared(floatX(np.asarray((np.random.randn(*(numHiddenNodes, numClasses))*.01))))

# Bias values
b_1 = theano.shared(value=numpy.zeros((featureMapsLayer1,), dtype=theano.config.floatX), borrow=True)
b_2 = theano.shared(value=numpy.zeros((featureMapsLayer2,), dtype=theano.config.floatX), borrow=True)
b_3 = theano.shared(value=numpy.zeros((featureMapsLayer3,), dtype=theano.config.floatX), borrow=True)
b_4 = theano.shared(value=numpy.zeros((numHiddenNodes,), dtype=theano.config.floatX), name='b', borrow=True)
b_5 = theano.shared(value=numpy.zeros((numHiddenNodes,), dtype=theano.config.floatX), name='b', borrow=True)
b_6 = theano.shared(value=numpy.zeros((numClasses,), dtype=theano.config.floatX), name='b', borrow=True)

params = [w_1, w_2, w_3, w_4, w_5, w_6, b_1, b_2, b_3, b_4, b_5, b_6]

## (2) Model
X = T.matrix('X').reshape((-1, 1, 96, 96)) # conv2d works with tensor4 type
Y = T.matrix('Y')

srng = RandomStreams()
def dropout(X, p=0.):
    if p > 0:
        X *= srng.binomial(X.shape, p=1 - p, dtype=theano.config.floatX)
        X /= 1 - p
    return X

# Theano provides built-in support for add convolutional layers
def model(X, w_1, w_2, w_3, w_4, w_5, w_6, p_1, p_2):
    l1 = dropout(T.tanh( max_pool_2d(T.maximum(conv2d(X, w_1, border_mode='full'),0.), (2, 2)) + b_1.dimshuffle('x', 0, 'x', 'x') ), p_1)
    l2 = dropout(T.tanh( max_pool_2d(T.maximum(conv2d(l1, w_2), 0.), (2, 2)) + b_2.dimshuffle('x', 0, 'x', 'x') ), p_1)
    l3 = dropout(T.flatten(T.tanh( max_pool_2d(T.maximum(conv2d(l2, w_3), 0.), (2, 2)) + b_3.dimshuffle('x', 0, 'x', 'x') ), outdim=2), p_1)# flatten to switch back to 1d layers
    l4 = dropout(T.maximum(T.dot(l3, w_4) + b_4, 0.), p_2)
    l5 = dropout(T.maximum(T.dot(l4, w_5) + b_5, 0.), p_2)
    return T.dot(l5, w_6) + b_6

y_hat_train = model(X, w_1, w_2, w_3, w_4, w_5, w_6, 0.2, 0.5)
y_hat_predict = model(X, w_1, w_2, w_3, w_4, w_5, w_6, 0., 0.)

## (3) Cost
cost = ((Y - (y_hat_predict))**2).mean()

## (4) Minimization.  
def backprop(cost, w, alpha=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=w)
    updates = []
    for w1, grad in zip(w, grads):
        
        # adding gradient scaling
        acc = theano.shared(w1.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * grad ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        grad = grad / gradient_scaling
        updates.append((acc, acc_new))
        
        updates.append((w1, w1 - grad * alpha))
    return updates

update = backprop(cost, params)
train = theano.function(inputs=[X, Y], outputs=cost, updates=update, allow_input_downcast=True)
y_pred = y_hat_predict
predict = theano.function(inputs=[X], outputs=y_pred, allow_input_downcast=True)

################################################################################
# Model Execution
################################################################################
all_X, all_Y = Load.load()

trX = all_X[:2800]
trY = all_Y[:2800]
deX = all_X[2800:]
deY = all_Y[2800:]

trX = trX.reshape(-1, 1, imageWidth, imageWidth)
deX = deX.reshape(-1, 1, imageWidth, imageWidth)

miniBatchSize = 1
def gradientDescentStochastic(epochs):
    trainTime = 0.0
    predictTime = 0.0
    start_time = time.time()
    for i in range(epochs):       
        for start, end in zip(range(0, len(trX), miniBatchSize), range(miniBatchSize, len(trX), miniBatchSize)):
            cost = train(trX[start:end], trY[start:end])
        pdeY = predict(deX)
        cost_de = ((deY - pdeY)**2).mean()
        print '%d) precision=%.8f, Traning cost=%.8f, DE cost: %.8f' %(i+1, np.mean(np.allclose(deY, pdeY)), cost, cost_de)
        trainTime =  trainTime + (time.time() - start_time)
    print 'train time = %.2f' %(trainTime)

gradientDescentStochastic(1)

print 'Mean squared error on Training data: %.8f\n'%((trY - trY.mean())**2).mean()
print 'Mean squared error on Dev data: %.8f\n'%((deY - deY.mean())**2).mean()


################################################################################
# Write to CSV
################################################################################
# Predict all development set
start_time = time.time()
pdeY = predict(deX)
print 'predict time = %.2f' %(time.time() - start_time)

# Write to csv
file_path = os.path.dirname(os.path.realpath(__file__)) + '/results.csv'
print ('Writing CSV file: {0}', file_path)

with open(file_path, 'wb') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=',') 
    array_str = str()
    for index in range(pdeY.shape[0]):
        serialized = pickle.dumps(pdeY[index], protocol=0) # protocol 0 is printable ASCII
        csv_writer.writerow([index, serialized])

sys.stdout.flush()
