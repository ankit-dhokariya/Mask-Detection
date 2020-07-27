import numpy as np


input_size = 14 * 14 * 8
h1_size = 1024
h2_size = 1024
h3_size = 512
num_labels = 2
lr = 0.001
reg = 0.5

def add_bias(A):

    X = np.ones((A.shape[0], 1))
    A = np.hstack((X, A))

    return A

def sigmoid(A):

    return 1. / (1. + np.exp(-A))

def sigmoidgradient(A):

    x = sigmoid(A)
    g = np.multiply(x, (1 - x))

    return g


def hypothesis(X, theta):

    h = X @ theta.T

    return h


class Logistic:

    def __init__(self):

        self.theta1 = np.random.rand(h1_size, input_size + 1) * (2 * 5) - 5
        self.theta2 = np.random.rand(h2_size, h1_size + 1) * (2 * 5) - 5
        self.theta3 = np.random.rand(h3_size, h2_size + 1) * (2 * 5) - 5
        self.theta4 = np.random.rand(num_labels, h3_size + 1) * (2 * 5) - 5

    def forward(self, input, label):

        m = input.shape[0]

        a1 = input.flatten().reshape(input.shape[0], input.shape[1] * input.shape[2] * input.shape[3])
        a1 = add_bias(a1)

        z2 = hypothesis(a1, self.theta1)
        a2 = sigmoid(z2)
        a2 = add_bias(a2)

        z3 = hypothesis(a2, self.theta2)
        a3 = sigmoid(z3)
        a3 = add_bias(a3)

        z4 = hypothesis(a3, self.theta3)
        a4 = sigmoid(z4)
        a4 = add_bias(a4)

        z5 = hypothesis(a4, self.theta4)
        a5 = sigmoid(z5)

        maxi = np.argmax(a5, axis=1)

        if num_labels == 1:
            out = np.around(a5)
        elif num_labels > 1:
            out = np.zeros(a5.shape)
            for i in range(m):
                out[i, maxi[i]] = 1

        if label is not None:
            cost = (a5 - label) ** 2
        else:
            cost = None

        return out, cost

    def backprop(self, input, label, lr, lam):

        m = input.shape[0]

        a1 = input.flatten().reshape(input.shape[0], input.shape[1] * input.shape[2] * input.shape[3])
        a1 = add_bias(a1)

        z2 = hypothesis(a1, self.theta1)
        a2 = sigmoid(z2)
        a2 = add_bias(a2)

        z3 = hypothesis(a2, self.theta2)
        a3 = sigmoid(z3)
        a3 = add_bias(a3)

        z4 = hypothesis(a3, self.theta3)
        a4 = sigmoid(z4)
        a4 = add_bias(a4)

        z5 = hypothesis(a4, self.theta4)
        a5 = sigmoid(z5)

        maxi = np.argmax(a5, axis=1)

        if num_labels == 1:
            out = np.around(a5)
        elif num_labels > 1:
            out = np.zeros(a5.shape)
            for i in range(m):
                out[i, maxi[i]] = 1

        cost = (a5 - label) ** 2

        d5 = a5 - label
        d4 = np.multiply((d5 @ self.theta4[:, 1:]), sigmoidgradient(z4))
        d3 = np.multiply((d4 @ self.theta3[:, 1:]), sigmoidgradient(z3))
        d2 = np.multiply((d3 @ self.theta2[:, 1:]), sigmoidgradient(z2))

        delta1 = d2.T @ a1
        delta2 = d3.T @ a2
        delta3 = d4.T @ a3
        delta4 = d5.T @ a4

        self.theta1[:][0] = 0
        self.theta2[:][0] = 0
        self.theta3[:][0] = 0

        theta1_grad = (delta1 / m) + ((lam) * self.theta1)
        theta2_grad = (delta2 / m) + ((lam) * self.theta2)
        theta3_grad = (delta3 / m) + ((lam) * self.theta3)
        theta4_grad = (delta4 / m) + ((lam) * self.theta4)

        self.theta1 -= lr * theta1_grad
        self.theta2 -= lr * theta2_grad
        self.theta3 -= lr * theta3_grad
        self.theta4 -= lr * theta4_grad

        return out, cost, sigmoidgradient(input)
