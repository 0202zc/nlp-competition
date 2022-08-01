import synonyms
from xpinyin import Pinyin
import pandas as pd
import numpy as np
import csv
import json

data = pd.read_csv(filepath_or_buffer='../test_A.csv', sep='	')
result_A = pd.read_csv('../vote.csv')

sentence1 = data.iloc[:, 0:1]
sentence2 = data.iloc[:, 1:2]

results = result_A.iloc[:, 0:1]

sentence1 = np.asarray(sentence1).tolist()
sentence2 = np.asarray(sentence2).tolist()
results = np.asarray(results).tolist()

count = 0

kw2similar_words = json.load(open("./data/kw2similar_words.json", "r", encoding='utf-8'))
keys_set = set(kw2similar_words.keys())

for i in range(0, 50000):
    pass
    # if synonyms.compare(sentence1[i][0], sentence2[i][0], True) >= 0.9:
    #     results[i][0] = 1
    #     count += 1
    #     print(str(i + 1) + " " + sentence1[i][0] + " " + sentence2[i][0])

print('测试集同义率：{}%；同义数：{}'.format(str(count / 50000 * 100), count))

with open('./result/ccf_qianyan_qm_result_A_synonyms.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    for row in results:
        writer.writerow(row)
