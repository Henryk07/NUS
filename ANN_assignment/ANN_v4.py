# NUS Chen Haolin
# ESp3201 Assignment 2 ANN_XOR
# libraries
from cgi import test
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from tqdm import tqdm


# Sigmoid Function [activation func]


def sigmoid(z):
    z = 1/(1+np.exp(-z))
    return z

# linear activation function


def linear(x):
    return x/1.15

# rectified linear function


def relu(x):
    return (np.maximum(0, x))

# initialize parameters


def initializeParameters(n_x, n_h, n_y):
    W1 = [[0.011246756  0.01857022]
          [0.0049142496 - 0.008718987]
          [0.006378179 - 0.002373877]]
    [[0.01785864 - 0.0048661778]
     [0.68447624  1.00665372]
     [-0.0095215288 - 0.003626722]]
    W2 = [[0.011246756  0.01857022]
          [0.0049142496 - 0.008718987]
          [0.006378179 - 0.002373877]]
    [[0.01785864 - 0.0048661778]
     [0.68447624  1.00665372]
     [-0.0095215288 - 0.003626722]]
    b1 = np.zeros((n_h, 1))
    b2 = 0.003

    parameters = {"W1": W1, "b1": b1,
                  "W2": W2, "b2": b2}
    return parameters

# Forward Propagation


def forwardPropagation(X, Y, parameters):
    m = X.shape[1]  # Total training examples
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2)
    # logprobs = np.multiply(np.log(A2), Y) + \
    #     np.multiply(np.log(1 - A2), (1 - Y))
    # loss = -np.sum(logprobs) / m
    #print('a', A2)
    #print('aaaaa', A2[0, :])
    # print(Y[1])
    losstest = 0
    #global losstest
    for j in range(4):
        losstest += np.square(A2[0, j]-Y[0]).sum()

    lossa = (-1/m) * np.sum(np.multiply(Y[0], np.log(A2[0, :])) +
                            np.multiply((1-Y[0]), np.log(1-A2[0, :])))

    lossb = (-1/m) * np.sum(np.multiply(Y[1], np.log(A2[1, :])) +
                            np.multiply((1-Y[1]), np.log(1-A2[1, :])))

    lossc = (-1/m) * np.sum(np.multiply(Y[2], np.log(A2[2, :])) +
                            np.multiply((1-Y[2]), np.log(1-A2[2, :])))

    lossd = (-1/m) * np.sum(np.multiply(Y[3], np.log(A2[3, :])) +
                            np.multiply((1-Y[3]), np.log(1-A2[3, :])))

    loss = (-1/m) * np.sum(np.multiply(Y, np.log(A2)) +
                           np.multiply((1-Y), np.log(1-A2)))
    # Make sure cost is a scalar
    loss = np.squeeze(loss)
    #lossa = np.squeeze(lossa)
    #lossb = np.squeeze(lossb)
    #lossc = np.squeeze(lossc)
    #lossd = np.squeeze(lossd)
    return losstest, lossd, lossc, lossb, lossa, loss, cache, A2

# Backward Propagation


def backwardPropagation(X, Y, cache):
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2) = cache

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, A1 * (1 - A1))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    gradients = {"dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    return gradients

# Updating the weights based on gradients


def updateParameters(parameters, gradients, l_r):
    parameters["W1"] = parameters["W1"] - l_r * gradients["dW1"]
    parameters["W2"] = parameters["W2"] - l_r * gradients["dW2"]
    parameters["b1"] = parameters["b1"] - l_r * gradients["db1"]
    parameters["b2"] = parameters["b2"] - l_r * gradients["db2"]
    return parameters


# XOR inputs
X = np.array([[1, 1, 0, 0],
              [0, 1, 0, 1]])
# The correct output of XOR
Y = np.array([1, 0, 0, 1])
# for each inputs
# Define model parameters
n_h = 3  # number of hidden layer neurons (3)
n_x = X.shape[0]  # number of input (2)
n_y = Y.shape[0]  # number of output(1)
# for each

parameters = initializeParameters(
    n_x, n_h, n_y)
parameters_each = initializeParameters(
    2, 3, 1)
epoch = 10  # training epoch setting
learningRate = 0.1  # learning rate
losses = np.zeros((epoch, 1))
lossa = np.zeros((epoch, 1))
lossb = np.zeros((epoch, 1))
lossc = np.zeros((epoch, 1))
lossd = np.zeros((epoch, 1))

pbar = tqdm(total=epoch)
for i in range(epoch):

    losstest, lossd[i, 0], lossc[i, 0], lossb[i, 0], lossa[i, 0], losses[i, 0], cache, A2 = forwardPropagation(
        X, Y, parameters)
    gradients = backwardPropagation(X, Y, cache)
    parameters = updateParameters(parameters, gradients, learningRate)
    pbar.update(1)

pbar.close()
# Evaluating the performance(loss value diagram)
plt.subplot(151)
plt.plot(lossa)
plt.subplot(152)
plt.plot(lossb)
plt.subplot(153)
plt.plot(lossc)
plt.subplot(154)
plt.plot(lossd)
plt.subplot(155)
plt.plot(losses)
#plt.suptitle('Categorical Plotting')
plt.xlabel("EPOCHS")
plt.ylabel("Loss value")
plt.show()
# plt.figure()
#plt.plot(loss_a, 'r--', losses, 'b')
# plt.xlabel("EPOCHS")
#plt.ylabel("Loss value")
# plt.show()

# Testing
X = np.array([[0, 1, 0, 1],
              [0, 1, 1, 0]])  # XOR test input
Y = np.array([0, 0, 1, 1])
_, _, _, _, loss, _, A2 = forwardPropagation(X, Y, parameters)
prediction = (A2 > 0.5) * 1.0
print(prediction[0, :])
# print("loss value is ", loss)

# if prediction == Y_t:
#   print("The input is", [X_t[0, i]
#        for i in X_t], [X_t[1, i] for i in X_t], "yes")
# else:
#   print("The input is", [X_t[0, i]
#          for i in X_t], [X_t[1, i] for i in X_t], "yes")
# print(A2)
