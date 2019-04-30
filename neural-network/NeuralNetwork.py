import numpy as np

# 双曲线函数


def tanh(x):
    return np.tanh(x)

# 双曲线函数导数


def tanh_derivative(x):
    return 1.0 - np.tanh(x)*np.tanh(x)

# 逻辑函数


def logistic(x):
    return 1/(1 + np.exp(-x))

# 逻辑函数导数


def logistic_derivative(x):
    return logistic(x) * (1-logistic(x))


class NeuralNetwork:

    # 构造函数
    # self = this
    def __init__(self, layers, activation='tanh'):
        """
        :param layers: A list containing the number of units in each layer Should be at least two values 每层里面有多少个神经元
        :param activation: The activation function to be used. Can be "logistic" or "tanh"
        """
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_derivative

        self.weights = []
        for i in range(1, len(layers) - 1):
            self.weights.append((2*np.random.random((layers[i-1] + 1, layers[i] + 1)) - 1)*0.25)  # i前面一层和i层
            self.weights.append((2*np.random.random((layers[i] + 1, layers[i + 1])) - 1)*0.25)  # i后面一层和i层

    # 训练 建模
    # 利用抽取样本的方法，没抽一次，来回更新权重一次，一次算一个epoch
    def fit(self, X, y, learning_rate=0.2, epochs = 10000):
        X = np.atleast_2d(X)
        temp = np.ones([X.shape[0], X.shape[1]+1])
        temp[:, 0:-1] = X # 添加偏向 bias
        X = temp
        y = np.array(y)

        for k in range(epochs):
            i = np.random.randint(X.shape[0]) # 随机抽样选一个x的实例
            a = [X[i]]

            for l in range(len(self.weights)):
                a.append(self.activation(np.dot(a[l], self.weights[l])))
            error = y[i] -a[-1]
            deltas = [error * self.activation_deriv(a[-1])]

            # start back propagation
            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))
            deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0] + 1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a
