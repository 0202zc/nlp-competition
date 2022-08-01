import numpy as np
import pandas as pd
import csv

data = pd.read_csv(filepath_or_buffer='../test_A.csv', sep='	')
sentence1 = np.asarray(data.iloc[:, 0:1]).tolist()
sentence2 = np.asarray(data.iloc[:, 1:2]).tolist()

res = []

for i in range(0, 50000):
    # if str(sentence1[i]).__contains__('航班') or str(sentence2[i]).__contains__('航班') or str(sentence2[i]).__contains__(
    #         '飞机') or str(sentence2[i]).__contains__('飞机') or (
    #         str(sentence2[i]).__contains__('飞') and str(sentence2[i]).__contains__('到')) or (str(
    #     sentence2[i]).__contains__('飞') and str(sentence2[i]).__contains__('到')):
    #     print(str(i + 1) + " " + sentence1[i][0] + " " + sentence2[i][0])
    # elif str(sentence1[i]).__contains__('火车') or str(sentence2[i]).__contains__('火车'):
    #     print(str(i + 1) + " " + sentence1[i][0] + " " + sentence2[i][0])
    # if (len(sentence1[i]) == len(sentence2[i]) and str(sentence1[i]).__contains__('高') and str(
    #         sentence2[i]).__contains__('矮') or str(sentence1[i]).__contains__('矮') and str(sentence2[i]).__contains__(
    #         '高')) and (str(sentence1[i]).__contains__('比') or str(sentence2[i]).__contains__('比')):
    #     print(str(i + 1) + " " + sentence1[i][0] + " " + sentence2[i][0])
    # elif str(sentence1[i]).__contains__('火车') or str(sentence2[i]).__contains__('火车'):
    #

    sorted_1 = sorted(sentence1[i][0])
    sorted_2 = sorted(sentence2[i][0])
    #
    # if len(sorted_1) == len(sorted_2):
    #     if sentence1[i][0].__contains__('比') and sentence2[i][0].__contains__('比') and (
    #             sentence1[i][0].__contains__('吗') or sentence2[i][0].__contains__('吗') or sentence1[i][0].__contains__(
    #         '么') or sentence2[i][0].__contains__('么')):
    #         print(str(i + 1) + " " + sentence1[i][0] + " " + sentence2[i][0])
        # if sorted_1.__contains__('高') and sorted_2.__contains__('矮') or sorted_2.__contains__(
        #     '高') and sorted_1.__contains__('矮'):
        #     print(str(i + 1) + " " + sentence1[i][0] + " " + sentence2[i][0])

    if sorted_1 == sorted_2:
        res.append(str(i + 1) + "," + sentence1[i][0] + "," + sentence2[i][0])
        print(str(i + 1) + " " + sentence1[i][0] + " " + sentence2[i][0])
    else:
        res.append(str(i + 1) + ",0,0")

with open('result/syntactic-structure.csv', 'w', newline='\n', encoding='utf8') as f:
    for i in range(0, len(res)):
        f.write(res[i] + '\n')
