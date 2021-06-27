import numpy


def sigmoid(sop):
    return 1.0 / (1 + numpy.exp(-1 * sop))


def error(predicted, target):
    return numpy.power(predicted - target, 2)


def error_predicted_derive(predicted, target):
    return 2 * (predicted - target)


def activation_sop_derive(sop):
    return sigmoid(sop) * (1.0 - sigmoid(sop))


def sop_w_derive(x):
    return x


def update_w(w, grad, learning_rate):
    return w - learning_rate * grad


x = 0.1
target = 0.3
learning_rate = 0.01
w = numpy.random.rand()
print("Initial W: ", w)

for k in range(1000000):
    y = w * x
    predicted = sigmoid(y)
    err = error(predicted, target)

    g1 = error_predicted_derive(predicted, target)
    g2 = activation_sop_derive(predicted)
    g3 = sop_w_derive(x)

    grad = g3 * g2 * g1
    print(predicted)

    w = update_w(w, grad, learning_rate)
