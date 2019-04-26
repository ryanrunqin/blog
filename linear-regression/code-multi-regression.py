from numpy import genfromtxt
import numpy as np
from sklearn import datasets, linear_model

# data.csv里的变量分别是 英里数 次数 车型 时间（Y）
data_path = r"F:\WebProjects\blog\linear-regression\data.csv"
dilivery_data = genfromtxt(data_path, delimiter=',')

# print("data")
# print(dilivery_data)

# 前面的':'表示所有行 后面的':-1'表示所有列但不包括-1（最后一列）
X = dilivery_data[:, :-1]
# 前面的':'表示所有行 后面的'-1'表示最后一列
Y = dilivery_data[:, -1]

regr = linear_model.LinearRegression()
# fit 对数据进行建模
regr.fit(X, Y)

# b1, b2, b3 ... bn
print("coefficients: {}".format(regr.coef_))

# 截距 b0
print("intercept: {}".format(regr.intercept_))

xPred = np.array([[102, 6]])
yPred = regr.predict(xPred)

print("predicted y: {}".format(yPred))
