from nodes import *

f = open('data/mnist_test_10.csv')
test_data = f.readlines()
f.close()


scoreboard = []
for data in test_data:
    # you can do this because the string acts as an iteration of characters before getting converted to a list
    # discovered on accident wew
    label = int(data[0])
    print(label)
    data = data[1:].split(',')
    results = n.query(np.asfarray(data) / 255 * 0.99 + 0.01)
    targets = np.zeros(outputnodes) + 0.01
    targets[label] = 0.99
    print(results.index(max(results)))
    if label == results.index(max(results)):
        scoreboard.append(1)
    else:
        scoreboard.append(0)
    final_score = sum(scoreboard)
    print(final_score)



