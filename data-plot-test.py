import matplotlib.pyplot as plt
import numpy as np
import csv

fig = plt.figure(figsize=(9, 4))
columns, rows = 5, 2

nlist = []
n = 0

with open('data/mnist_test_10.csv') as f:
    reader = csv.reader(f)

    for row in reader:
        # matplotlib stuff
        n += 1
        fig.add_subplot(rows, columns, n)

        # will be turned into a numpy array
        vals = []

        # checks if it's the first item so <if> only iterates once,
        # to do calculations without opening file again or being destructive
        if reader.line_num == 1:
            # checks how many items the row has (skipping 1st, which is assigned number for the data)
            # items correspond to pixels
            area = len(row) - 1
            # it's a square 2D image, so we can determine width and height using square root
            size = int((np.sqrt(area)))

        number = row[0]
        for column in row[1:]:
            # basically just reads from csv and appends to vals
            vals.append(float(column))

        vals = np.array(vals)
        vals = np.reshape(vals, [size, size])
        plt.imshow(vals, cmap='gray_r')

    f.close()

plt.show()

