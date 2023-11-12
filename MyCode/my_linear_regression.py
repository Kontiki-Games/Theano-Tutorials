import theano
from theano import tensor as T
from theano import shared
from theano import function
import numpy as np

    # 1. 生成训练样本
# 1.a 在 [-1, 1] 的区间内，用 f(x) = 2x 生成样本点
# 1.b 用服从正态分布的随机数去扰动上一步的函数值
trainX = np.linspace(-1, 1, 101)
trainY = 2.0 * trainX + np.random.randn(*trainX.shape) * 0.33
#print(trainX)
#print(trainY)

#for x, y in zip(trainX, trainY):
#    print(x, y, "\n")


# 2. 构建模型，损失函数，训练函数
# 2.- 定义模型函数, 截距为0的线性函数
def model(x, k):
    return k * x

# 2.a 创建输入和输出符号(训练样本)
X = T.scalar("X")
Y = T.scalar("Y")

# 2.b 创建模型参数
w = shared(np.asarray(0.0, dtype=theano.config.floatX))

# 2.c 创建模型表达式
y = model(X, w)

# 2.d 创建损失函数表达式
cost = T.mean( T.sqr(y - Y) )

# 2.e 计算 cost 关于 w 的梯度
gradient = T.grad(cost=cost, wrt=w)

# 2.f w 更新
updates = [(w, w - 0.01 * gradient)]

# 2.g 定义训练函数
train = function(inputs = [X, Y], outputs = cost, updates = updates, allow_input_downcast=True)

# 3. 训练
for i in range(100):
    for x, y in zip(trainX, trainY):
        train(x, y)
    
print(w.get_value())