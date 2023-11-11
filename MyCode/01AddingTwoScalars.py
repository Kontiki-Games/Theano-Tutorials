import numpy
import theano.tensor as T
from theano import function
x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
f = function([x, y], z)
r = f(2, 3)
print(r)
r = numpy.allclose(f(16.3, 12.1), 28.4)
print(r)