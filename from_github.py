import numpy as np
from scipy.special import expit as sigmoid
from scipy.special import logit as inv_sigmoid

import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, n_of_nodes, learning_rate):
        # list of number of nodes for each layer, going through input to output
        self.nodes_n = n_of_nodes
        # first layer (inputs) should have no weights
        self.weights = []

        self.outputs = []

        inodes = None
        for nodes in self.nodes_n:
            # skips input nodes, we can't assign weights to them since they represent raw input
            onodes = nodes
            if inodes:
                # generates array with random numbers of shape inode lines x onodes columns
                node_weights = np.random.normal(0, pow(inodes, -0.5), (onodes, inodes))
                # self.weights will have same amounts of items on n_of_nodes minus 1
                self.weights.append(node_weights)
            inodes = nodes

        # learning rate
        self.lr = learning_rate


    def query(self, input_list):
        # converts list to 1D array
        inputs = np.array(input_list)
        # acts as if it was first layer's output
        self.outputs = [inputs]

        # for array of available node weights; (for a 3 layer network, it would be 2)
        for node_weights in self.weights:
            # current layer output is dot product of weights and previous output
            # selects last output from outputs list, as they get appended
            output = np.dot(node_weights, self.outputs[-1])
            output = sigmoid(output)
            self.outputs.append(np.array(output))

        # once there are no more weights layer, returns final output
        return output

    def train(self, input_list, target_list):
        # error margin is target value - output value
        # total error to be divided between network node weights

        errors = []
        # layer number is.. well, the amount of layers
        layer = len(self.nodes_n)
        print(layer)

        while layer > 1:
            layer -= 1
            if layer == len(self.nodes_n) - 1:
                error = target_list - np.array(self.query(input_list), ndmin=2)
            else:
                error = np.dot(self.weights[layer].T, error)
            errors.append(error)

        for error in errors:
            layer -= 1
            self.weights[layer] += self.lr * np.dot((error * (1 - error)).T, np.expand_dims(self.outputs[layer], axis=0))


# ------ INPUT ------

# n of nodes
# inputnodes: total of pixels from database set (28x28)
# hiddennodes: -
# outputnodes: all possible outputs (numbers from 0 to 9)
inputnodes, hiddennodes, outputnodes = 784, 200, 10

# learning rate
lrate = 0.1

# epoch: repetition of training session
epoch = 1

# - INITIALISATION --
n = NeuralNetwork([inputnodes, hiddennodes, outputnodes], lrate)

# ------ DATA -------

train_data = []

f = open('data/mnist_train_100.csv')
for line in f:
    train_data.append(line.split(','))
f.close()

# ---- TRAINING ----

for iteration in range(epoch):
    for data in train_data:
        # np.asfarray() is same as array() but automatically converts strings into numbers
        # reshape tuple: [int(np.sqrt(len(values[1:])))] * 2
        inputs = np.asfarray(data[1:])
        # arrays can be done math operations with, which apply to every individual value in the array
        # inputs must NOT have value 0.00, because it can kill weights when multiplied with other weights (n * 0 = 0)
        inputs = (inputs / 255 * 0.99) + 0.01

        # create target 1D list with same amount of numbers as outputnodes
        targets = np.zeros(outputnodes) + 0.01
        # first items in value[] is the number label. the specified index is then the max output value (0.99)
        # (differs from input list because the input can be 100%, but the network cannot be 100% sure on an output)
        # values[] index must be kept at 0; dataset gives the number's label to the first value
        targets[int(data[0])] = 0.99
        n.train(inputs, targets)

# ---- TESTING -----

test_data = []
f = open('data/mnist_test_10.csv')
for line in f:
    test_data.append(f.readline().split(','))
f.close()


scoreboard = []
for data in test_data:
    label = int(data[0])
    results = n.query(np.asfarray(data[1:]) / 255 * 0.99 + 0.01)
    if label == np.argmax(results):
        scoreboard.append(1)
    else:
        scoreboard.append(0)
print('accuracy | ', (sum(scoreboard) / len(scoreboard)))
