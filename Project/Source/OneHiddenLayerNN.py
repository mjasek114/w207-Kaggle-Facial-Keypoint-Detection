import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
import time
from Data import Load


srng = RandomStreams()

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(name, shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01), name)

def rectify(X):
    return T.maximum(X, 0.)

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0., 'acc')
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def sgd(cost, w, lr=0.01):
    grads = T.grad(cost=cost, wrt=w)
    updates = []
    for w1, grad in zip(w, grads):
        updates.append([w1, w1 - grad * lr])
    return updates

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def model(X, w_h, w_o, p_drop_input, p_drop_hidden):
    X = dropout(X, p_drop_input)

    h = rectify(T.dot(X, w_h))
    h = dropout(h, p_drop_hidden)

    py_x = rectify(T.dot(h, w_o))
    return h, py_x


allX, allY = Load.load()
trX = allX[:1800]
trY = allY[:1800]
deX = allX[1800:]
deY = allY[1800:]


X = T.fmatrix("X")
Y = T.fmatrix("Y")

w_h = init_weights("w_h", (9216, 100))
w_o = init_weights("w_o", (100, 30))
w_o_printed = theano.printing.Print('Weights on output layer')(w_o)

noise_h, noise_py_x = model(X, w_h, w_o, 0.2, 0.5)
h, py_x = model(X, w_h, w_o, 0., 0.)

xent = (Y - noise_py_x)**2 # L2 distance for cost
cost = xent.mean() # The cost to minimize (MSE)

params = [w_h, w_o]
updates = sgd(cost, params, lr=0.01)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=py_x, allow_input_downcast=True)

miniBatchSize = 1
def gradientDescentStochastic(epochs):
    trainTime = 0.0
    predictTime = 0.0
    start_time = time.time()
    for i in range(epochs):       
        for start, end in zip(range(0, len(trX), miniBatchSize), range(miniBatchSize, len(trX), miniBatchSize)):
            cost = train(trX[start:end], trY[start:end])
        print '%d) precision=%.4f, cost=%.4f' %(i+1, np.mean(np.allclose(deY, predict(deX))), cost)
        trainTime =  trainTime + (time.time() - start_time)
    print 'train time = %.2f' %(trainTime)

gradientDescentStochastic(400)    