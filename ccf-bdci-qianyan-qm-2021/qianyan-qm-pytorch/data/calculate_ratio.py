import pandas as pd
import numpy as np

data = pd.read_csv('big_training_set.csv', header=None, names=['sentence1', 'sentence2', 'label'], sep='\t')

sentence1 = data.iloc[:, 0:1]
sentence2 = data.iloc[:, 1:2]
results = np.asarray(data.iloc[:, 2:3]).tolist()

all = len(results)
positive = 0
negative = 0

for i in range(all):
    if results[i][0] == 1:
        positive += 1

negative = all - positive

print('正样本比例：{}，负样本比例：{}'.format(str(positive / all), str(negative / all)))
