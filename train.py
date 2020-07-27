import numpy as np
import cv2
import pickle
import argparse
from imutils import paths
import matplotlib.pyplot as plt
from test import test
from Conv3x3 import Conv3x3
from MaxPool import MaxPool2
from Logistic import Logistic

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-m", "--model", type=str, required=True,
                help="path to trained model")
args = vars(ap.parse_args())

input_size = 14 * 14 * 8
h1_size = 1024
h2_size = 1024
h3_size = 512
num_labels = 2
batch_size = 2000
epochs = 1
lr = 0.003
reg = 0.5

path_train = args["dataset"] + "\\training"
path_test = args["dataset"] + "\\testing"
X_train = []
Y_train = []
X_test = []
Y_test = []
Epoch = []
error = []
mask_face = [1, 0]
no_mask = [0, 1]

print("[INFO] Labeling Images...\n...")

for imagePath in paths.list_images(path_train):
    img = cv2.imread(imagePath, 0)
    X_train.append(img)
    label = imagePath.split("\\")[-2]
    if label == "mask_face":
        Y_train.append(mask_face)
    else:
        Y_train.append(no_mask)

for imagePath in paths.list_images(path_test):
    img = cv2.imread(imagePath, 0)
    X_test.append(img)
    label = imagePath.split("\\")[-2]
    if label == "mask_face":
        Y_test.append(mask_face)
    else:
        Y_test.append(no_mask)

print("[INFO] Images Labeled")
X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)
print("\nShape of X_train =", X_train.shape)
print("Shape of Y_train =", Y_train.shape)

m = X_train.shape[0]

print("\n[INFO] Initializig CNN...\n...")
conv = Conv3x3(8)
pool = MaxPool2()
nn = Logistic()
print("[INFO] CNN Initialized")

plt.figure(1)

print("\n[INFO] Training...")
for e in range(epochs):
    step = []
    step_error = []
    m_cost = 0
    Epoch.append((e + 1))
    plot_label = "Epoch%d" % (e + 1)
    total_cost = 0
    print("--- Epoch %d ---" % (e + 1))
    permutation = np.random.permutation(len(X_train))
    X_train = X_train[permutation]
    Y_train = Y_train[permutation]

    for i in range((m // batch_size) + 1):
        acc = 0
        temp_m = 0
        if (m // batch_size) == i:
            inputs = X_train[(i * batch_size):]
            labels = Y_train[(i * batch_size):]
            temp_m = inputs.shape[0]
        else:
            inputs = X_train[(i * batch_size):(i * batch_size) + batch_size]
            labels = Y_train[(i * batch_size):(i * batch_size) + batch_size]
            temp_m = inputs.shape[0]

        out = conv.forward((inputs / 255) - .5)
        out = pool.forward(out)
        out, cost, gradient = nn.backprop(out, labels, lr, reg)
        gradient = pool.backprop(gradient)
        gradient = conv.backprop(gradient, lr)

        for j in range(temp_m):
            if (out[j] == labels[j]).all():
                acc += 1

        total_cost += np.sum(cost) / num_labels
        avg_cost = np.sum(cost) / (num_labels * temp_m)
        for s in range(temp_m):
            m_cost += np.sum(cost[s])
            step_error.append(m_cost)
            step.append(i * batch_size + s)

        print("[STEP %d] Past %d steps Average cost = %.8f | Accuracy = %.2f%%" % ((i * batch_size) + temp_m, temp_m, avg_cost, (acc / temp_m) * 100))

    step = np.asarray(step)
    step_error = np.asarray(step_error)
    plt.plot(step, step_error, label=plot_label)
    plt.xlabel('m')
    plt.ylabel('error')

    error.append(total_cost / m)

plt.legend()
plt.show()

plt.figure(2)
Epoch = np.asarray(Epoch)
error = np.asarray(error)
plt.plot(Epoch, error)
plt.xlabel('epoch')
plt.ylabel('error')
plt.show()
print("[INFO] Model Trained")

# Save Weights
print("\n[INFO] Saving...")
weights = np.array([conv.filters, nn.theta1, nn.theta2, nn.theta3, nn.theta4])
with open(args["model"], 'wb') as f:
    pickle.dump(weights, f)
print("[INFO] Model Saved")

print("\n[INFO] Testing...")

X_test = np.asarray(X_test)
Y_test = np.asarray(Y_test)
print("\nShape of X_test =", X_test.shape)
print("Shape of Y_test =", Y_test.shape)

inputs = X_test
labels = Y_test.reshape(Y_test.shape[0], num_labels)

test(args["model"], inputs, labels)
