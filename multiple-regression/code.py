# python 实现相关度和 R平方(决定系数)
import numpy as np
from astropy.units import Ybarn
import math

# 计算皮尔逊相关系数
def computeCorrelation(X, Y):
    xBar = np.mean(X)
    yBar = np.mean(Y)
    SSR = 0
    varX = 0
    varY = 0
    for i in range(0, len(X)):
        diffXvsXbar = X[i] - xBar
        diffYvsYbar = Y[i] - yBar
        SSR += (diffXvsXbar * diffYvsYbar)
        varX += diffXvsXbar**2
        varY += diffYvsYbar**2

    SST = math.sqrt(varX * varY)
    return SSR / SST



# degree --次. x^2就是二次 这里是一次 没有x平方的值
def polyfix(x,y,degree):
    results= {}
    coeffs = np.polyfit(x, y, degree)

    # results['polynomial'] 包含斜率和截距，一个方程的必要参数
    results['polynomial'] = coeffs.tolist() #字典to list

    # r squared
    p = np.poly1d(coeffs) #一维线性回归 这个p就是估计的理想方程
    yhat = p(x)
    ybar = np.sum(y)/len(y) # np.mean(Y)
    ssreg = np.sum((yhat-ybar)**2)
    sstotal = np.sum((y-ybar)**2)
    results['determination'] = ssreg / sstotal
    return results




textX = [1,3,8,7,9]
textY = [10,12,24,21,34]

print ("r: {}".format(computeCorrelation(textX, textY)))
print ("r^2: {}".format(computeCorrelation(textX, textY)**2))

print ("r^2: {}".format(polyfix(textX, textY, 1)['determination']))