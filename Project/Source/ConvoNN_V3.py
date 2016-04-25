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
from sklearn.utils import shuffle

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d

################################################################################
# Global utilities and functions
################################################################################
srng = RandomStreams()

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def dropout(X, p=0.):
    if p > 0:
        X *= srng.binomial(X.shape, p=1 - p, dtype=theano.config.floatX)
        X /= 1 - p
    return X

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

################################################################################
## (1) Convolutional Network Parameters
################################################################################
numClasses = 30
numHiddenNodes = 1000 
patchWidth = 3
patchHeight = 3
featureMapsLayer1 = 32
featureMapsLayer2 = 64
featureMapsLayer3 = 128

# Regularization parameter for the fully connected NN
lambda_reg = .00000001

# For convonets, we will work in 2d rather than 1d.  The dataset images are 96 * 96
imageWidth = 96

# Convolution layers.  
w_1 = theano.shared(floatX(np.asarray((np.random.randn(*(featureMapsLayer1, 1, patchWidth, patchHeight))*.01))))
w_2 = theano.shared(floatX(np.asarray((np.random.randn(*(featureMapsLayer2, featureMapsLayer1, patchWidth, patchHeight))*.01))))
w_3 = theano.shared(floatX(np.asarray((np.random.randn(*(featureMapsLayer3, featureMapsLayer2, patchWidth, patchHeight))*.01))))

# Fully connected NN. 
w_4 = theano.shared(floatX(np.asarray((np.random.randn(*(featureMapsLayer3 * 10 * 10, numHiddenNodes))*.01))))
w_5 = theano.shared(floatX(np.asarray((np.random.randn(*(numHiddenNodes, numHiddenNodes))*.01))))
w_6 = theano.shared(floatX(np.asarray((np.random.randn(*(numHiddenNodes, numClasses))*.01))))

# Bias values
b_1 = theano.shared(value=numpy.zeros((featureMapsLayer1,), dtype=theano.config.floatX), borrow=True)
b_2 = theano.shared(value=numpy.zeros((featureMapsLayer2,), dtype=theano.config.floatX), borrow=True)
b_3 = theano.shared(value=numpy.zeros((featureMapsLayer3,), dtype=theano.config.floatX), borrow=True)

params = [w_1, w_2, w_3, w_4, w_5, w_6, b_1, b_2, b_3]

## (2) Model
X = T.matrix('X').reshape((-1, 1, 96, 96)) # conv2d works with tensor4 type
Y = T.matrix('Y')


# Theano provides built-in support for add convolutional layers
def convolutional_model(X, w_1, w_2, w_3, w_4, w_5, w_6, p_1, p_2, p_3, p_4, p_5):
    l1 = dropout(T.tanh( max_pool_2d(T.maximum(conv2d(X, w_1, border_mode='full'),0.), (2, 2),ignore_border=True) + b_1.dimshuffle('x', 0, 'x', 'x') ), p_1)
    l2 = dropout(T.tanh( max_pool_2d(T.maximum(conv2d(l1, w_2), 0.), (2, 2),ignore_border=True) + b_2.dimshuffle('x', 0, 'x', 'x') ), p_2)
    l3 = dropout(T.flatten(T.tanh( max_pool_2d(T.maximum(conv2d(l2, w_3), 0.), (2, 2),ignore_border=True) + b_3.dimshuffle('x', 0, 'x', 'x') ), outdim=2), p_3)# flatten to switch back to 1d layers
    l4 = dropout(T.maximum(T.dot(l3, w_4), 0.), p_4)
    l5 = dropout(T.maximum(T.dot(l4, w_5), 0.), p_5)
    return T.dot(l5, w_6)

y_hat_train = convolutional_model(X, w_1, w_2, w_3, w_4, w_5, w_6, 0.1, 0.2, 0.3, 0.5, 0.5)
y_hat_predict = convolutional_model(X, w_1, w_2, w_3, w_4, w_5, w_6, 0., 0., 0., 0., 0.)

## (3) Regularized Cost
cost = ((Y - (y_hat_predict))**2).mean() + lambda_reg*((w_4**2).sum() + (w_5**2).sum() + (w_6**2).sum())

## (4) Minimization.  
update = backprop(cost, params)
train = theano.function(inputs=[X, Y], outputs=cost, updates=update, allow_input_downcast=True)
y_pred = y_hat_predict
predict = theano.function(inputs=[X], outputs=y_pred, allow_input_downcast=True)

###############################################################################
# One Hidden layer NN to process convo prediction output
###############################################################################
back_num_inputs = 9
back_num_hidden = 30
back_num_outputs = 30
back_pdrop_input = .0
back_pdrop_hidden = .0

back_X = T.fmatrix("back_X")
back_Y = T.fmatrix("back_Y")

back_w_h = theano.shared(floatX(np.asarray((np.random.randn(*(back_num_inputs, back_num_hidden))*.01))))
back_w_o = theano.shared(floatX(np.asarray((np.random.randn(*(back_num_hidden, back_num_outputs))*.01))))

def back_model(back_X, back_w_h, back_w_o, p_drop_input, p_drop_hidden):
    X = dropout(back_X, p_drop_input)

    h = T.maximum(T.dot(back_X, back_w_h), 0.)
    h = dropout(h, back_pdrop_hidden)

    back_py_x = T.dot(h, back_w_o)
    return h, back_py_x


back_noise_h, back_noise_py_x = back_model(back_X, back_w_h, back_w_o, back_pdrop_input, back_pdrop_hidden)
back_h, back_py_x = back_model(back_X, back_w_h, back_w_o, 0., 0.)


back_cost = ((back_Y - back_noise_py_x)**2).mean()

back_params = [back_w_h, back_w_o]
back_updates = backprop(back_cost, back_params)

back_train = theano.function(inputs=[back_X, back_Y], outputs=back_cost, updates=back_updates, allow_input_downcast=True)
back_predict = theano.function(inputs=[back_X], outputs=back_py_x, allow_input_downcast=True)


################################################################################
# Model Execution
################################################################################

all_X, all_Y = Load.load()

test_indices = np.random.choice(all_X.shape[0], .2*all_X.shape[0], replace=False)
keep = np.ones(all_X.shape[0], dtype=bool) # array of True matching 1st dim
keep[test_indices] = False
trX = all_X[keep,:]
trY = all_Y[keep,:]
deX = all_X[test_indices]
deY = all_Y[test_indices]

rot_X, rot_Y = Load.rotate_images(trX, trY)
blur_X, blur_Y = Load.blurr_images(trX, trY)
trans_X, trans_Y = Load.transpose_images(trX, trY)

trX, trY = np.vstack((trX, rot_X)), np.vstack((trY, rot_Y))
trX, trY = np.vstack((trX, blur_X)), np.vstack((trY, blur_Y))
trX, trY = np.vstack((trX, trans_X)), np.vstack((trY, trans_Y))

trX, trY = shuffle(trX, trY, random_state=42)  # shuffle data
rot_X, rot_Y, blur_X, blur_Y, tr_X, tr_Y = None, None, None, None, None, None

trX = trX.reshape(-1, 1, imageWidth, imageWidth)
deX = deX.reshape(-1, 1, imageWidth, imageWidth)

miniBatchSize = 1
def convolutional_gradientDescentStochastic(epochs):
    training_costs=[]
    dev_costs=[]
    trainTime = 0.0
    predictTime = 0.0
    start_time = time.time()
    for i in range(epochs):       
        for start, end in zip(range(0, len(trX), miniBatchSize), range(miniBatchSize, len(trX), miniBatchSize)):
            cost = train(trX[start:end], trY[start:end])
        
        # Compute overall dev and training costs after epoch
        pdeY = predict(deX)
        cost_de = ((deY - pdeY)**2).mean()
        
        ptrY = predict(trX)
        cost_tr = ((trY - ptrY)**2).mean()
        
        # Capture the traning and dev costs on each epoch
        training_costs.append(cost_tr)
        dev_costs.append(cost_de)
        
        print '%d) Convo precision=%.8f, Traning cost=%.8f, DE cost: %.8f' %(i+1, np.mean(np.allclose(deY, pdeY)), cost_tr, cost_de)
        trainTime =  trainTime + (time.time() - start_time)
    print 'train time = %.2f' %(trainTime)
    
    return training_costs, dev_costs, ((trY - ptrY)**2), ((deY - pdeY)**2)

convo_training_costs, convo_dev_costs,  convo_training_costs_raw, convo_dev_costs_raw = convolutional_gradientDescentStochastic(1)


back_miniBatchSize = 1
def back_gradientDescentStochastic(epochs):
    training_costs=[]
    dev_costs=[]
    trainTime = 0.0
    predictTime = 0.0
    start_time = time.time()
    for i in range(epochs):       
        for start, end in zip(range(0, len(convo_training_costs_raw), miniBatchSize), range(miniBatchSize, len(convo_training_costs_raw), miniBatchSize)):
            cost = back_train(convo_training_costs_raw[start:end], trY[start:end])

        # Compute overall dev and training costs after epoch
        pdeY = back_predict(convo_dev_costs_raw)
        cost_de = ((deY - pdeY)**2).mean()

        ptrY = back_predict(trX)
        cost_tr = ((trY - ptrY)**2).mean()

        # Capture the traning and dev costs on each epoch
        training_costs.append(cost_tr)
        dev_costs.append(cost_de)

        print '%d) Back precision=%.4f, Traning cost=%.4f, DE cost: %.4f' %(i+1, np.mean(np.allclose(deY, pdeY)), cost_tr, cost_de)
        trainTime =  trainTime + (time.time() - start_time)
    print 'train time = %.2f' %(trainTime)

    return training_costs, dev_costs

back_training_costs, back_dev_costs = back_gradientDescentStochastic(1)


print 'Mean squared error on Training data: %.8f\n'%((trY - trY.mean())**2).mean()
print 'Mean squared error on Dev data: %.8f\n'%((deY - deY.mean())**2).mean()