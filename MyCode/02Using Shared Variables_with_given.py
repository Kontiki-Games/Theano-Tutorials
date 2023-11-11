import theano
from theano import shared 
from theano import tensor as T
from theano import function

state = shared(0)
inc = T.iscalar('inc')

fn_of_state = 2 * state + inc

foo = T.scalar(dtype = state.dtype)

skip_shared = function([inc, foo], fn_of_state, givens=[(state, foo)])

print(skip_shared(1, 3))
print(state.get_value())

