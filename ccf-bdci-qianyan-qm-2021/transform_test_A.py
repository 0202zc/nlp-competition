from xpinyin import Pinyin
import Levenshtein as L
import pandas as pd
import numpy as np
import csv

p = Pinyin()
data = pd.read_csv(filepath_or_buffer='test_A.csv', sep='	')
# result_A = pd.read_csv('origin_batch128_ernie_gram.csv')
result_A = pd.read_csv('corrector/result/90_499.csv', header=None, names=['label'])

sentence1 = data.iloc[:, 0:1]
sentence2 = data.iloc[:, 1:2]

results = result_A.iloc[:, 0:1]

sentence1 = np.asarray(sentence1).tolist()
sentence2 = np.asarray(sentence2).tolist()
results = np.asarray(results).tolist()
count = 0
all_count = 0


def is_misspelling(s1, s2):
    sentence_1 = p.get_pinyin(s1)
    sentence_2 = p.get_pinyin(s2)
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
        return True
    return False


# print('测试集同音率：{}%；同音数：{}'.format(str(count / 50000 * 100), count))
all_count += count
count = 0

for i in range(50000):
    if str(sentence1[i][0]).__contains__('×') or str(sentence2[i][0]).__contains__('×'):
        results[i][0] = 0
        count += 1

print('"×" 符号共有%s个' % str(count))
all_count += count
count = 0

for i in range(50000):
    s1 = str(sentence1[i][0])
    s2 = str(sentence2[i][0])

    if len(s1) == len(s2):
        for j in range(len(s1)):
            if is_misspelling(s1, s2[j:] + s2[:j]):
                # if s1 == s2[j:] + s2[:j]:
                print(str(s1) + '\t' + str(s2) + '\t' + str(results[i][0]))
                count += 1
                results[i][0] = 1
                break

print('sorted相同共有%s个' % str(count))
all_count += count
print('共修改%s个标签' % str(all_count))
# origin_batch128_ernie_gram.csv

with open('./result/origin_batch128_ernie_gram.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    for row in results:
        writer.writerow(row)

# with open('result/test_A_90_499.csv', 'w', newline='\n', encoding='utf8') as f:
#     for i in range(0, 50000):
#         f.write(sentence1[i][0] + '\t' + sentence2[i][0] + '\t' + str(results[i][0]) + '\n')
