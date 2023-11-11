import theano
import numpy as np

def func(a):
	print(a)

trX = np.linspace(-1, 1, 101)
func(trX.shape) 