import theano
from theano import tensor as T
import numpy as np

# -- 1. 生成样本
trX = np.linspace(-1, 1, 101)
trY = 2 * trX + np.random.randn(*trX.shape) * 0.33
# * 这里实参 *trX.shape，是将 (100,) 这个tuple，转为了， 100
# * np.random.randn: 生成 n 个符合标准正态分布的样本点
# * https://numpy.org/doc/stable/reference/random/generated/numpy.random.randn.html

# -- 2. 构建损失函数的表达式。
# 其中输入是每个样本的输入和输出，float类型，非shared（个人称为 in）
# 模型参数为 w，float类型，shared（个人称为 inout）
# 模型表达式为 y。
# 损失函数表达式为 cost
# 更新列表为可解释为： w = w - gradient * 0.01
X = T.scalar()
Y = T.scalar()

def model(X, w):
    return X * w

w = theano.shared(np.asarray(0., dtype=theano.config.floatX))
# * asarray() 可以将结构性数据转为 ndarray。包括 lists, lists of tuples, tuples, tuples of tuples, tuples of lists and ndarrays。 
# * 如果数据已经是ndarray就不会有copy行为。这是和array()的不同之处.
# * asarray(a, dtype = None, order=None, *, like=None). a: array like, dtype: data type. 通常由a推导出来。
# * shared 函数构建 shared变量（符号）。可以认为是 inout 性质的函数参数。
y = model(X, w)

cost = T.mean(T.sqr(y - Y))
gradient = T.grad(cost=cost, wrt=w)
updates = [[w, w - gradient * 0.01]]

# -- 3. 训练
train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
# * updates 是一张包含若干pair的表，pair.first是一个shared变量，pair.second是一个表达式。每次执行函数后，将用表达式的值更新shared变量。
# * 关于 allow_input_downcast=True 的解释： https://stackoverflow.com/questions/45573293/what-does-allow-input-downcast-do-in-theano

for i in range(100):
    for x, y in zip(trX, trY):
        train(x, y)
        
print(w.get_value()) #something around 2

