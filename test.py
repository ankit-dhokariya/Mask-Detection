import pickle
from Conv3x3 import Conv3x3
from MaxPool import MaxPool2
from Logistic import Logistic


def test(model, inputs, labels):

    cv_m = inputs.shape[0]
    acc = 0

    conv = Conv3x3(8)
    pool = MaxPool2()
    nn = Logistic()

    with open(model, 'rb') as f:
        weights = pickle.load(f)

    conv.filters = weights[0]
    nn.theta1 = weights[1]
    nn.theta2 = weights[2]
    nn.theta3 = weights[3]
    nn.theta4 = weights[4]

    out = conv.forward((inputs / 255) - .5)
    print("\nConv3x3 output shape =", out.shape)
    out = pool.forward(out)
    print("MaxPool output shape =", out.shape)
    out, cost = nn.forward(out, labels)
    avg_cost = sum(cost) / cv_m
    for j in range(cv_m):
        if (out[j] == labels[j]).all():
            acc += 1
    print("Test Cost = %.5f | Accuracy = %.2f%% \n" % (avg_cost[0], (acc / cv_m) * 100))
