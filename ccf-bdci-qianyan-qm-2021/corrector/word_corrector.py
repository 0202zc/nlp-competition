from xpinyin import Pinyin

import pycorrector
import Levenshtein as L
import pandas as pd
import numpy as np
import csv

p = Pinyin()
data = pd.read_csv(filepath_or_buffer='../test_A.csv', sep='	')
sentence1 = np.asarray(data.iloc[:, 0:1]).tolist()
sentence2 = np.asarray(data.iloc[:, 1:2]).tolist()
result_A = pd.read_csv('../kfold_csv/result/weight_fusion.csv')

results = np.asarray(result_A.iloc[:, 0:1]).tolist()

count = 0

for i in range(50000):
    # print(sentence1[i][0] + '\t' + sentence2[i][0])
    # print(str(i + 1) + '\t' + sentence1[i][0] + '\t' + str(pycorrector.correct(sentence1[i][0])[0]))
    sentence1[i][0] = str(pycorrector.correct(sentence1[i][0])[0])
    sentence2[i][0] = str(pycorrector.correct(sentence2[i][0])[0])
    print(str(i + 1))
    sentence_1 = p.get_pinyin(sentence1[i][0])
    sentence_2 = p.get_pinyin(sentence2[i][0])

    is_nasal = False
    if abs(len(sentence_2) - len(sentence_1)) == 1:
        for j in range(min(len(sentence_1), len(sentence_2))):
            if sentence_1[j] != sentence_2[j]:
                break
        s_1 = sentence_1[::-1]
        s_2 = sentence_2[::-1]
        for k in range(min(len(s_1), len(s_2))):
            if s_1[k] != s_2[k]:
                break
        min_len = min(len(sentence_1), len(sentence_2))
        if ((j == min_len - k or j >= min_len - 1 or k >= min_len - 1) and (
                sentence_1[j] == 'g' or sentence_1[k] == 'g' or sentence_2[j] == 'g' or sentence_2[k] == 'g')):
            is_nasal = True

    if sentence_1 == sentence_2 or sentence_1.replace("l", "n") == sentence_2 or sentence_1.replace("n",
                                                                                                    "l") == sentence_2 or (
            len(sentence_1) == len(sentence_2) and L.hamming(sentence_1, sentence_2) < 1):
        results[i][0] = 1
        count += 1

print('测试集同音率：{}%；同音数：{}'.format(str(count / 50000 * 100), count))

# ccf_qianyan_qm_resultA.csv

with open('result/pycorrector_misspelling_result_A.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    for row in results:
        writer.writerow(row)
