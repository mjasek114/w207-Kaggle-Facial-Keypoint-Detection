#%matplotlib inline

import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
import time
import matplotlib
from matplotlib import pyplot

from Data import Load

srng = RandomStreams()

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(name, shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01), name)

def rectify(X):
    return T.maximum(X, 0.)

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

    py_x = T.dot(h, w_o)
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

noise_h, noise_py_x = model(X, w_h, w_o, 0.2, 0.5)
h, py_x = model(X, w_h, w_o, 0., 0.)

cost = ((Y - noise_py_x)**2).mean()

params = [w_h, w_o]
updates = sgd(cost, params, lr=.01)

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
        pdeY = predict(deX)
        cost_de = ((deY - pdeY)**2).mean()
        print '%d) precision=%.4f, Traning cost=%.4f, DE cost: %.4f' %(i+1, np.mean(np.allclose(deY, pdeY)), cost, cost_de)
        trainTime =  trainTime + (time.time() - start_time)
    print 'train time = %.2f' %(trainTime)

gradientDescentStochastic(10) 

print 'Mean squared error on Training data: %.4f\n'%((trY - trY.mean())**2).mean()
print 'Mean squared error on Dev data: %.4f\n'%((deY - deY.mean())**2).mean()


'''
#
# Let's see how the predictions come out
#
def plot_sample(x, y, y_pred, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y_pred[1::2] * 48 + 48, marker='x', s=10)
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', c='r', s=10)

y_pred = predict(deX)

fig = pyplot.figure(figsize=(6, 6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)


for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    plot_sample(deX[i], deY[i], y_pred[i], ax)

pyplot.show()
'''