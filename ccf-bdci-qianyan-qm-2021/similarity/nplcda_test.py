import numpy as np
import pandas as pd

from nlpcda import Randomword  # 随机(等价)实体替换
from nlpcda import Similarword  # 随机同义词替换

data = pd.read_csv(filepath_or_buffer='../test_A.csv', sep='	')
result_A = pd.read_csv('../vote.csv')

sentence1 = data.iloc[:, 0:1]
sentence2 = data.iloc[:, 1:2]
results = result_A.iloc[:, 0:1]

sentence1 = np.asarray(sentence1).tolist()
sentence2 = np.asarray(sentence2).tolist()
results = np.asarray(results).tolist()


def random_word(word):
    smw = Randomword(create_num=3, change_rate=0.3)
    return smw.replace(word)


def similar_word(word):
    smw = Similarword(create_num=3, change_rate=0.3)
    return smw.replace(word)


# test_str = "时空猎人的密码怎么改	时空猎人好被怎么改密码"
# smw = Randomword(create_num=3, change_rate=0.3)
# rs1 = smw.replace(test_str)
#
# print('随机实体替换>>>>>>')
# for s in rs1:
#     print(s)
#
# smw = Similarword(create_num=3, change_rate=0.3)
# rs1 = smw.replace(test_str)
#
# print('随机同义词替换>>>>>>')
# for s in rs1:
#     print(s)

# https://github.com/425776024/nlpcda

if __name__ == '__main__':
    for i in range(0, 50001):
        s = sentence1[i][0] + '	' + sentence2[i][0]
        res = similar_word(s)
        print('%s>>>>>>随机同义词替换>>>>>>%s' % (s, res))
