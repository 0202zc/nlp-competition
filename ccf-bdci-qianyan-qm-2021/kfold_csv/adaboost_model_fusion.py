import numpy as np
import pandas as pd

K = 5

weights = (np.asarray(pd.read_csv('source/new/weights.csv', header=None, names=['weight'])).transpose()[0]).tolist()
sum_weight = 0
for i in range(K):
    sum_weight += weights[i]

k_fold = []
for i in range(K):
    fold = (np.asmatrix(pd.read_csv('source/new/fold{}_misspelling.csv'.format(str(i + 1)), header=None, names=['label']).iloc[:, 0:1]).transpose()[0]).tolist()
    k_fold.append(fold[0])


def activation(x):
    return 0 if x <= 0.5 else 1


def weight_average():
    global sum_weight
    global k_fold

    result = []

    temp = np.empty([1, 100000], dtype=float)
    for i in range(K):
        temp += np.dot(weights[i], k_fold[i])

    for i in range(100000):
        result.append(activation(temp[0][i] / sum_weight))

    return result


if __name__ == '__main__':
    data = weight_average()

    with open(file='result/weight_fusion.csv', mode='w', encoding='utf8', newline='\n') as f:
        for item in data:
            f.write(str(item) + '\n')
