import theano
from theano import shared 
from theano import tensor as T
from theano import function

state = shared(0)
inc = T.iscalar('inc')
accumulator = function([inc], state, updates=[(state, state+inc)])

print(state.get_value())

accumulator(1)
print(state.get_value())

accumulator(300)
print(state.get_value())

state.set_value(-1)
print(state.get_value())

decrementor = function([inc], state, updates=[(state, state-inc)])

decrementor(2)
print(state.get_value())

