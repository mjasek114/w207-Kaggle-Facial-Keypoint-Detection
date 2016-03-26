import numpy as np
import theano.tensor as T
from theano import function
from theano import pp
from theano import shared
from theano import config
from theano.tensor.shared_randomstreams import RandomStreams

# Example 1 - scalars
x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
f = function([x, y], z)

print f(2, 3)
print np.allclose(f(16.3, 12.1), 28.4)
print pp(z)

# Example 2 - mnatric addition
# dmatrix is the type of matrix of double
x = T.dmatrix('x')
y = T.dmatrix('y')
z = x + y
f = function([x, y], z)
print f([[1, 2], [3, 4]], [[10, 20], [30, 40]])

# Example 3 Basic tensor functionality
x = T.fmatrix()

print (False,)*5
dtensor5 = T.TensorType('float64', (False,)*5)
x = dtensor5()

print np.random.randn(3,4)
x = shared(np.random.randn(3,4))

a = T.tensor4()
b = T.tensor4()
c = T.tensor4()
x = T.stack([a, b, c])
x.ndim # x is a 5d tensor.
rval = x.eval(dict((t, np.zeros((2, 2, 2, 2))) for t in [a, b, c]))
print rval.shape


print np.arange(9)
n = np.arange(9).reshape(3,3)
print n
print n[n > 4]

t = T.arange(9).reshape((3,3))
print t.eval()
print t[(t > 4)].eval()
print (t > 4).nonzero()
print t[(t > 4).nonzero()].eval()

# Computation of the logic function memberwise on the elements of an array of doubles
x = T.dmatrix('x')
s = 1 / (1 + T.exp(-x))
logistic = function([x], s)
print logistic([[0, 1], [-1, -2]])

# Alternate formulation using tanh function
s2 = (1 + T.tanh(x / 2)) / 2
logistic2 = function([x], s2)
print logistic2([[0, 1], [-1, -2]])

# Binary Matrix operations
a, b = T.dmatrices('a', 'b')
diff = a - b
abs_diff = abs(diff)
diff_squared = diff**2
f = function([a, b], [diff, abs_diff, diff_squared])
f([[1, 1], [1, 1]], [[0, 1], [2, 3]])

# Setting a default value for an argument
from theano import In
x, y = T.dscalars('x', 'y')
z = x + y
f = function([x, In(y, value=1)], z)
print f(33)
print f(33, 2)

# Shared variables examples
state = shared(0)
inc = T.iscalar('inc')
accumulator = function([inc], state, updates=[(state, state+inc)])

print(state.get_value())
accumulator(1)
print(state.get_value())
accumulator(300)
print(state.get_value())

# Set the value of the shared
state.set_value(-1)
accumulator(3)
print(state.get_value())

decrementor = function([inc], state, updates=[(state, state-inc)])
decrementor(2)
print(state.get_value())

# Using a replacement for a shared variable in a function
fn_of_state = state * 2 + inc

# The type of foo must match the shared variable we are replacing
# with the ``givens``
foo = T.scalar(dtype=state.dtype)
skip_shared = function([inc, foo], fn_of_state, givens=[(state, foo)])
skip_shared(1, 3)  # we're using 3 for the state, not state.value
print(state.get_value()) 

# Using random number generators
srng = RandomStreams(seed=234)
rv_u = srng.uniform((2,2))
rv_n = srng.normal((2,2))
f = function([], rv_u)
g = function([], rv_n, no_default_updates=True)    #Not updating rv_n.rng
nearly_zeros = function([], rv_u + rv_u - 2 * rv_u)

f_val0 = f()
f_val1 = f()
g_val0 = g()  # different numbers from f_val0 and f_val1
g_val1 = g()

print f_val0
print f_val1
print g_val0
print g_val1

# As usual for shared variables, the random number generators used for random variables
# are common between functions. So our nearly_zeros function will update the state of 
# the generators used in function f above.
state_after_v0 = rv_u.rng.get_value().get_state()
nearly_zeros()       # this affects rv_u's generator
v1 = f()
rng = rv_u.rng.get_value(borrow=True)
rng.set_state(state_after_v0)
rv_u.rng.set_value(rng, borrow=True)
v2 = f()             # v2 != v1
v3 = f()             # v3 == v1

N = 50
rng = np.random
print rng.randint(size=N, low=0, high=2)