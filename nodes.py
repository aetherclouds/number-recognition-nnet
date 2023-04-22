import numpy as np
from scipy.special import expit as sigmoid
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):

        # n of nodes for input, hidden, output layers
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes

        # learning rate
        self.lr = learning_rate

        # weights
        self.wih = np.random.normal(0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

    def train(self, input_list, target_list):
        # NOTE: find a way to reutilise self.query() without returning final_output and being able to use the function vars.
        # converts inputs to 2D array (2D image)
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate final signals from hidden layer
        hidden_outputs = sigmoid(hidden_inputs)

        # calculate signals into hidden layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate final signals from hidden layer
        final_outputs = sigmoid(final_inputs)

        output_errors = targets - final_inputs
        hidden_errors = np.dot(self.who.T, output_errors)

        self.who += self.lr * np.dot(output_errors * final_outputs * (1 - final_outputs), np.transpose(hidden_outputs))

        self.wih += self.lr * np.dot(hidden_errors * hidden_outputs * (1 - hidden_outputs), np.transpose(inputs))

    def query(self, input_list):
        # converts inputs to 2d array (2D image)
        inputs = np.array(input_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate final signals from hidden layer
        hidden_outputs = sigmoid(hidden_inputs)

        # calculate signals into hidden layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate final signals from hidden layer
        final_outputs = sigmoid(final_inputs)

        return final_outputs


# ------ INPUT ------

# n of nodes
# inputnodes: total of pixels from database set (28x28)
# hiddennodes: -
# outputnodes: all possible outputs (numbers from 0 to 9)
inputnodes, hiddennodes, outputnodes = 784, 100, 10
# learning rate
lrate = 0.3

# -- INITIALIZATION -
n = NeuralNetwork(inputnodes, hiddennodes, outputnodes, lrate)

# ------ DATA -------

data = []

f = open('data/mnist_train.csv')
for line in f:
    data.append(f.readline().split(','))
f.close()

for record in data:
    # np.asfarray() is same as array() but automatically converts strings into numbers
    # reshape tuple: [int(np.sqrt(len(values[1:])))] * 2
    inputs = np.asfarray(record[1:])
    # arrays can be done math operations with, which apply to every individual value in the array
    # inputs must NOT have value 0.00, because it can kill weights when multiplied with other weights (n * 0 = 0)
    inputs = inputs / 255 * 0.99 + 0.01

    # create target 1D list with same amount of numbers as outputnodes
    targets = np.zeros(outputnodes) + 0.01
    # first items in value[] is the number label. the specified index is then the max output value (0.99)
    # (differs from input list because the input can be 100%, but the network cannot be 100% sure on an output)
    # values[] index must be kept at 0; dataset gives the number's label to the first value
    targets[int(record[0])] = 0.99
    n.train(inputs, targets)

f = open('data/mnist_test.csv')
test_data = f.readlines()
f.close()


scoreboard = []
for data in test_data:
    data = data.split(',')
    label = int(data[0])
    results = n.query(np.asfarray(data[1:]) / 255 * 0.99 + 0.01)
    targets = np.zeros(outputnodes) + 0.01
    targets[label] = 0.99
    # np.where(): same as list.index() if <results> was a list.
    # above is pointless because you can just use np.argmax() which returns index of max number on an array. rip
    if np.argmax(targets) == np.argmax(results):
        scoreboard.append(1)
    else:
        scoreboard.append(0)
print((sum(scoreboard) / len(scoreboard)))
