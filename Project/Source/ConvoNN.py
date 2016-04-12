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

all_X, all_Y = Load.load()

################################################################################
# Model classes and utility functions
################################################################################

# Implement dropout using a binomial distribution according to the parameters of the call
class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), p_drop =.2):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape#,
            #input_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = dropout(T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')), p_drop)

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

# Implement statistical dropout
def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

# Implement rectifier activation
def rectify(X):
    return T.maximum(X, 0.)

# Initialize the weights of our NN using a normal distribution.
def init_weights(name, shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01), name)

# Stochastic gradient descent
def sgd(cost, w, lr=0.01):
    grads = T.grad(cost=cost, wrt=w)
    updates = []
    for w1, grad in zip(w, grads):
        updates.append([w1, w1 - grad * lr])
    return updates

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh, p_drop=.5):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        
        if W is None:
            W = init_weights("w_h", (n_in, n_out))

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
            
        self.W = W
        self.b = b
        self.params = [self.W, self.b]
        
        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            dropout(lin_output, p_drop) if activation is None
            else dropout(activation(lin_output), p_drop)
        )
        
class RegressionLayer(object):
    """Multi-feature Linear Regression Class

    The linear regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a least cost fit.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the linear regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = init_weights("w_o", (n_in, n_out))
        
        
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
        
        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.y_pred = T.dot(input, self.W) + self.b

        # parameters of the model
        self.params = [self.W, self.b]
        #self.params = [self.W]

        # keep track of model input
        self.input = input
    
    def cost(self, y):
        return ((y - self.y_pred)**2).mean()
        
        
################################################################################
# Model Instantiation
################################################################################
x = T.matrix('x')   # the data is presented as rasterized images
y = T.matrix('y')  # the labels are presented as a matrix of
            # facial feature locations

# Dropout probablitites through the NN
p_drop_convo = .2
p_drop_back_nn = .5

# Learning rate
learning_rate = .001

# Define the minibatch size
miniBatchSize = 1

# Define the number of kernels on layer 0 and 1 of the convolution
nkerns=[20, 50]

rng = numpy.random.RandomState(1234)
srng = RandomStreams()

######################
# BUILD ACTUAL MODEL #
######################
print('... building the model')

# Reshape matrix of rasterized images of shape (miniBatchSize, 96 * 96)
# to a 4D tensor, compatible with our LeNetConvPoolLayer
# (96, 96) is the size of MNIST images.
#layer0_input = x.reshape((miniBatchSize, 1, 96, 96))
layer0_input = x.reshape((-1, 1, 96, 96))

# Construct the first convolutional pooling layer:
# filtering reduces the image size to (96-9+1 , 96-9+1) = (88, 88)
# maxpooling reduces this further to (88/2, 88/2) = (44, 44)
# 4D output tensor is thus of shape (miniBatchSize, nkerns[0], 88, 88)
layer0 = LeNetConvPoolLayer(
    rng,
    input=layer0_input,
    image_shape=(miniBatchSize, 1, 96, 96),
    filter_shape=(nkerns[0], 1, 9, 9),
    poolsize=(2, 2),
    p_drop = p_drop_convo
)

# Construct the second convolutional pooling layer
# filtering reduces the image size to (44-9+1, 44-9+1) = (36, 36)
# maxpooling reduces this further to (36/2, 26/2) = (18, 18)
# 4D output tensor is thus of shape (miniBatchSize, nkerns[1], 18, 18)
layer1 = LeNetConvPoolLayer(
    rng,
    input=layer0.output,
    image_shape=(miniBatchSize, nkerns[0], 44, 44),
    filter_shape=(nkerns[1], nkerns[0], 9, 9),
    poolsize=(2, 2),
    p_drop = p_drop_convo
)

# the HiddenLayer being fully-connected, it operates on 2D matrices of
# shape (miniBatchSize, num_pixels) (i.e matrix of rasterized images).
# This will generate a matrix of shape (miniBatchSizebatch_size, nkerns[1] * 18 * 18),
# or (500, 50 * 18 * 18) = (500, 16200) with the default values.
layer2_input = layer1.output.flatten(2)

# construct a fully-connected sigmoidal layer
layer2 = HiddenLayer(
    rng,
    input=layer2_input,
    n_in=nkerns[1] * 18 * 18,
    n_out=500,
    activation=rectify, p_drop = p_drop_back_nn
)

# classify the values of the fully-connected sigmoidal layer
layer3 = RegressionLayer(input=layer2.output, n_in=500, n_out=30)

# the cost we minimize during training is the NLL of the model
cost = layer3.cost(y)

# create a list of all model parameters to be fit by gradient descent
params = layer3.params + layer2.params + layer1.params + layer0.params

# create a list of gradients for all model parameters
grads = T.grad(cost, params)

updates = sgd(cost, params, lr=learning_rate)

train = theano.function(inputs=[x, y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[x], outputs=layer3.y_pred, allow_input_downcast=True)


################################################################################
# Model Execution
################################################################################
trX = all_X[:1800]
trY = all_Y[:1800]
deX = all_X[1800:]
deY = all_Y[1800:]

def gradientDescentStochastic(epochs):
    trainTime = 0.0
    predictTime = 0.0
    start_time = time.time()
    for i in range(epochs):       
        for start, end in zip(range(0, len(trX), miniBatchSize), range(miniBatchSize, len(trX), miniBatchSize)):
            cost = train(trX[start:end], trY[start:end])
        pdeY = predict(deX)
        cost_de = ((deY - pdeY)**2).mean()
        print '%d) precision=%.4f, Traning cost=%.4f, DE cost: %.4f' %(i+1, np.mean(np.allclose(deY, pdeY)), cost, cost_de)
        trainTime =  trainTime + (time.time() - start_time)
    print 'train time = %.2f' %(trainTime)

gradientDescentStochastic(1000) 

print 'Mean squared error on Training data: %.4f\n'%((trY - trY.mean())**2).mean()
print 'Mean squared error on Dev data: %.4f\n'%((deY - deY.mean())**2).mean()


################################################################################
# Write to CSV
################################################################################
# Predict all development set
pdeY = predict(deX)

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
