import pandas as pd
import numpy as np
import pypinyin

# from xpinyin import Pinyin

# p = Pinyin()
data = pd.read_csv(filepath_or_buffer='test_b/test_B_1118.tsv', sep='	')
sentence1 = np.asarray(data.iloc[:, 0:1]).tolist()
sentence2 = np.asarray(data.iloc[:, 1:2]).tolist()
# result_A = pd.read_csv('../kfold_csv/result/weight_fusion.csv', header=None, names=['label'])
# result_A = pd.read_csv('../result_A_89_897.csv', header=None, names=['label'])
result_A = pd.read_csv('test_b/model_89_897_testB.csv', header=None, names=['label'])

results = np.asarray(result_A.iloc[:, 0:1]).tolist()


def check_synonyms(s1, s2):
    if len(s1) != len(s2):
        return False
    for i in range(len(s1)):
        if len(s1[i]) == 1 and len(s2[i]) == 1:
            if s1[i][0] != s2[i][0]:
                return False
        elif len(s1[i]) > 1 and len(s2[i]) == 1:
            if not (s2[i][0] in s1[i]):
                return False
        elif len(s2[i]) > 1 and len(s1[i]) == 1:
            if not (s1[i][0] in s2[i]):
                return False
        else:
            flag = False
            for j in range(len(s1[i])):
                if s1[i][j] in s2[i]:
                    flag = True
                    break
            if not flag:
                return False
    return True


count = 0
output = []

for i in range(100000):
    stc1 = str(sentence1[i][0])
    stc2 = str(sentence2[i][0])

    if str(sentence1[i][0]).__contains__('×') or str(sentence2[i][0]).__contains__('×'):
        results[i][0] = 0
        output.append(str(stc1) + '\t' + str(stc2) + '\t' + str(results[i][0]))
    else:
        s1 = [pypinyin.pinyin(word, heteronym=True, style=pypinyin.STYLE_NORMAL)[0] for word in sentence1[i][0]]
        # s2 = [pypinyin.pinyin(word, heteronym=True, style=pypinyin.STYLE_NORMAL)[0] for word in sentence2[i][0]]

        if len(stc1) == len(stc2):
            for j in range(len(stc1)):
                temp = stc2[j:] + stc2[:j]
                s2 = [pypinyin.pinyin(word, heteronym=True, style=pypinyin.STYLE_NORMAL)[0] for word in temp]
                if check_synonyms(s1, s2):
                    # print(str(stc1) + '\t' + str(stc2) + '\t' + str(results[i][0]))
                    count += 1
                    output.append(str(stc1) + '\t' + str(stc2) + '\t' + str(results[i][0]))
                    results[i][0] = 1
                    break

    # if check_synonyms(s1, s2):
    #     # print(str(i + 1) + "\t" + sentence1[i][0] + "\t" + sentence2[i][0])
    #     results[i][0] = 1
    #     count += 1

print('测试集同音率：{}%；同音数：{}'.format(str(count / 100000 * 100), count))

for i in range(100000):
    if str(sentence1[i][0]).__contains__('×') or str(sentence2[i][0]).__contains__('×'):
        results[i][0] = 0

with open('result/test_b/synonyms_list.txt', 'w', encoding='utf8', newline='\n') as f:
    for row in output:
        f.write(row + '\n')

with open(file='result/pinyin_misspelling_result_B.csv', mode='w', newline='\n', encoding='utf8') as f:
    for result in results:
        f.write(str(result[0]) + '\n')
