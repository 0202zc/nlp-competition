import numpy as np
import random
import pandas as pd

from misspelling_generate import cut_sentence


def generate(sentence, k):
    # 得到分词列表seg
    seg = cut_sentence(sentence)
    results = []
    for j in range(k):
        print(seg)
        if len(seg) > 1:
            num1, num2 = random.sample(range(0, len(seg)), 2)
            result = ''
            for i in range(len(seg)):
                if i != num1 and i != num2:
                    result += seg[i]
            result += seg[num1] + seg[num2]

            if not results.__contains__(result):
                results.append(result)

    return results


def main(sentence, k):
    res = generate(sentence, k)
    results = []

    for item in res:
        results.append('{}\t{}\t1'.format(sentence, item))

    return results


if __name__ == '__main__':
    # results = main('a9x大小', 1)
    # for result in results:
    #     print(result)

    results = []

    data = pd.read_csv(filepath_or_buffer='test_A_89_750.csv', sep='	')

    sentence1 = np.asarray(data.iloc[:, 0:1]).tolist()
    sentence2 = np.asarray(data.iloc[:, 1:2]).tolist()
    # label = np.asarray(data.iloc[:, 2:3]).tolist()

    for i in range(50000):
        if i % 2 is 0:
            generate_sentence = main(sentence1[i][0], 1)
        else:
            generate_sentence = main(sentence2[i][0], 1)
        results.append(generate_sentence)

    with open('result/generate_conversational.csv', 'w', encoding='utf8', newline='\n') as f:
        for row in results:
            if len(row) != 0:
                f.write(row[0] + '\n')
