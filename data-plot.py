import numpy as np
import matplotlib.pyplot as plt

f = open('data/mnist_test_10.csv')
data = f.readlines()
f.close()

values = data[0].split(',')
# np.asfarray() is same as array() but automatically converts strings into numbers
# reshape tuple: [int(np.sqrt(len(values[1:])))] * 2
array = np.asfarray(values[1:]).reshape((28, 28))
# arrays can be done math operations with, which apply to every individual value in the array
array = array / 255 * 0.99 + 0.01
print(array)
plt.imshow(array, cmap='gray_r', interpolation='None')
plt.show()