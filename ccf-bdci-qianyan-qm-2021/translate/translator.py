import pandas as pd
import numpy as np
import random

from translate import Translator


def convert_from_microsoft(text):
    translator1 = Translator(from_lang='chinese', to_lang='english')
    translation1 = translator1.translate(text)

    translator2 = Translator(from_lang='english', to_lang='chinese')
    translation2 = translator2.translate(translation1)

    if translation2 != text:
        return translation2

    return 'same'


data = pd.read_csv('../test_train.csv', sep='	')
sentence1 = data.iloc[:, 0:1]
sentence2 = data.iloc[:, 1:2]
label = data.iloc[:, 2:3]

sentence1 = np.asarray(sentence1).tolist()
sentence2 = np.asarray(sentence2).tolist()
label = np.asarray(label).tolist()

covert_list = []
file_length = len(sentence1)
cnt = 0

for i in range(0, file_length):
    row = ''
    conv_1 = convert_from_microsoft(sentence1[i][0])
    conv_2 = convert_from_microsoft(sentence2[i][0])
    if conv_1 is not 'same':
        row += conv_1
    else:
        row += sentence1[i][0]
    row += '	'
    if conv_2 is not 'same':
        row += conv_2
    else:
        row += sentence2[i][0]
    row += ('	' + str(label[i][0]))
    covert_list.append(row)
    print(row)
    step = random.choice([1, 3, 59, 500, 27064, 1506])
    i += step
    cnt += 1
    if cnt > (file_length / 2):
        break

with open('./result/translate_result.csv', newline='\n', encoding='utf8') as f:
    for row in covert_list:
        f.write(row)
