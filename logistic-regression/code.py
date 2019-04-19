import numpy as np
import random

# 梯度下降算法
# alpha 学习率
# m 总共实例数
# numIterations 循环次数
# 解决最小值
def gradientDescent(x, y, theta, alpha, m, numIterations):
    xTrans = x.transpose()
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta) # 内积
        loss = hypothesis - y # hypothesis is y_hat
        cost = np.sum(loss ** 2) / (2 * m)
        print("Iterations %d / Cost: %f" % (i, cost))

        gradient = np.dot(xTrans, loss) / m
        theta = theta - alpha * gradient
    return theta

# 生成数据

# 骗值， 方差
def genData(numPoints, bias, variance):
    x = np.zeros(shape=(numPoints, 2)) # numPoints行， 2列
    y = np.zeros(shape=numPoints) # 归类的标签
    for i in range(0, numPoints):
        x[i][0] = 1
        x[i][1] = i
        y[i] = (i + bias) + random.uniform(0,1) * variance
    return x, y

x, y = genData(100, 25, 10)
# print ("x: {}".format(x))
# print ("y: {}".format(y))

numIterations = 100000
alpha = 0.05
m, n = np.shape(x)
theta = np.ones(n)
theta = gradientDescent(x, y, theta, alpha, m, numIterations)
print(theta)