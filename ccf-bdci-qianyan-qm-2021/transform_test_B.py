from xpinyin import Pinyin
import pandas as pd
import numpy as np
import csv

p = Pinyin()
data = pd.read_csv(filepath_or_buffer='test_B_1118.tsv', sep='	')
# result_A = pd.read_csv('origin_batch128_ernie_gram.csv')
result_A = pd.read_csv('corrector/result/pinyin_misspelling_result_B.csv', header=None, names=['label'])

sentence1 = data.iloc[:, 0:1]
sentence2 = data.iloc[:, 1:2]

results = result_A.iloc[:, 0:1]

sentence1 = np.asarray(sentence1).tolist()
sentence2 = np.asarray(sentence2).tolist()
results = np.asarray(results).tolist()
count = 0
all_count = 0

with open('result/test_B_merge.csv', 'w', newline='\n', encoding='utf8') as f:
    for i in range(0, 100000):
        f.write(sentence1[i][0] + '\t' + sentence2[i][0] + '\t' + str(results[i][0]) + '\n')

data = pd.read_csv('result/test_B_merge.csv', sep='\t')
data.to_csv("test_B_merge.csv", index=False, sep='\t')
