import numpy as np

# x is list
# y is list
# SLR -- Simple Linear Regression


def fitSLR(x, y):
    n = len(x)
    numerator = 0  # 分子
    denominator = 0  # 分母
    for i in range(0, n):
        numerator += (x[i] - np.mean(x)) * (y[i] - np.mean(y))
        denominator += (x[i] - np.mean(x)) ** 2

    b1 = numerator / float(denominator)
    b0 = np.mean(y) / float(np.mean(x))

    return b0, b1


def predict(x, b0, b1):
    return b0 + x*b1


x = [1, 3, 2, 1, 3]
y = [14, 24, 18, 17, 27]

b0, b1 = fitSLR(x, y)

print("b0: {}".format(b0))
print("b1: {}".format(b1))

x_test = 6
y_test = predict(x_test, b0, b1)

print("y_test: {}".format(y_test))
