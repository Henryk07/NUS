
from cProfile import label
from cgi import test
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from tqdm import tqdm

# These are XOR inputs
x = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
# These are XOR outputs(Correct)
y = np.array([[0, 1, 1, 0]])
# Number of inputs
n_x = 2
# Number of neurons in hidden layer
n_h = 3
# Number of neurns in output layer
n_y = 1
# Total training examples
m = x.shape[1]
# Learning rate
lr = 0.1
# Define random seed for consistent results
np.random.seed(2)
# Define weight matrices for neural network
w1 = np.random.rand(n_h, n_x)   # Weight matrix for hidden layer
w2 = np.random.rand(n_y, n_h)   # Weight matrix for output layer
# I didnt use bias units
# We will use this list to accumulate losses
losses = []
loss_aa = []
loss_bb = []
loss_cc = []
loss_dd = []

# I used sigmoid activation function for hidden layer and output


def mean_squared_error(actual, predicted):
    sum_square_error = 0.0
    for i in range(1):
        sum_square_error += (actual - predicted)**2.0
    mean_square_error = 1.0 / 1 * sum_square_error
    return mean_square_error


def sigmoid(z):
    z = 1/(1+np.exp(-z))
    return z

# Forward propagation


def forward_prop(w1, w2, x):
    z1 = np.dot(w1, x)
    a1 = sigmoid(z1)
    z2 = np.dot(w2, a1)
    a2 = sigmoid(z2)
    return z1, a1, z2, a2

# Backward propagation


def back_prop(m, w1, w2, z1, a1, z2, a2, y):

    dz2 = a2-y
    dw2 = np.dot(dz2, a1.T)/m
    dz1 = np.dot(w2.T, dz2) * a1*(1-a1)
    dw1 = np.dot(dz1, x.T)/m
    dw1 = np.reshape(dw1, w1.shape)

    dw2 = np.reshape(dw2, w2.shape)
    return dz2, dw2, dz1, dw1


epoch = 50000  # 50000
pbar = tqdm(total=epoch)
for i in range(epoch):
    z1, a1, z2, a2 = forward_prop(w1, w2, x)
    loss_a = (a2[0, 0] - y[0, 0])**2
    loss_b = (a2[0, 1] - y[0, 1])**2
    loss_c = (a2[0, 2] - y[0, 2])**2
    loss_d = (a2[0, 3] - y[0, 3])**2
    loss = -(1/m)*np.sum(y*np.log(a2)+(1-y)*np.log(1-a2))
    losses.append(loss)
    loss_aa.append(loss_a)
    loss_bb.append(loss_b)
    loss_cc.append(loss_c)
    loss_dd.append(loss_d)
    pbar.update(1)
    da2, dw2, dz1, dw1 = back_prop(m, w1, w2, z1, a1, z2, a2, y)
    w2 = w2-lr*dw2
    w1 = w1-lr*dw1

pbar.close()


def predict(w1, w2, input):
    z1, a1, z2, a2 = forward_prop(w1, w2, test)
    a2 = np.squeeze(a2)
    if a2 >= 0.5:
        # ['{:.2f}'.format(i) for i in x])
        print("For input", [i[0] for i in input], "output is 1")
    else:
        print("For input", [i[0] for i in input], "output is 0")


# We plot losses to see how our network is doing


plt.figure()
line1, = plt.plot(losses, label="Total loss value")
line2, = plt.plot(loss_aa, label="loss value for input (0,0)")
line3, = plt.plot(loss_bb, label="loss value for input (0,1)")
line4, = plt.plot(loss_cc, label="loss value for input (1,0)")
line5, = plt.plot(loss_dd, label="loss value for input (1,1)")
leg = plt.legend(loc='upper right')
plt.xlabel("EPOCHS")
plt.ylabel("Loss value")
plt.title("Loss value against Epochs")
plt.show()


test = np.array([[1], [0]])
predict(w1, w2, test)
test = np.array([[1], [1]])
predict(w1, w2, test)
test = np.array([[0], [0]])
predict(w1, w2, test)
test = np.array([[0], [1]])
predict(w1, w2, test)
