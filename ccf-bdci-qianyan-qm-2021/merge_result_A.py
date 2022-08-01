from xpinyin import Pinyin
import pandas as pd
import numpy as np

p = Pinyin()
data = pd.read_csv(filepath_or_buffer='test_A.csv', sep='	')
# result_A = pd.read_csv('origin_batch128_ernie_gram.csv')
result_A = pd.read_csv('kfold_csv/result/weight_fusion.csv', header=None, names=['label'])

sentence1 = data.iloc[:, 0:1]
sentence2 = data.iloc[:, 1:2]

results = result_A.iloc[:, 0:1]

sentence1 = np.asarray(sentence1).tolist()
sentence2 = np.asarray(sentence2).tolist()
results = np.asarray(results).tolist()

# 合并A榜测试集和预测标签
with open('result/origin_batch128_ernie_gram.csv', 'w', newline='\n', encoding='utf8') as f:
    for i in range(0, 50000):
        f.write(sentence1[i][0] + '\t' + sentence2[i][0] + '\t' + str(results[i][0]) + '\n')

print('Finished!')
