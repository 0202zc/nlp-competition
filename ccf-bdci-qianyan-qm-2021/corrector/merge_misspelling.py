import pandas as pd
import numpy as np

result_1 = np.asarray(pd.read_csv('result/pycorrector_misspelling_result_A.csv')).transpose().tolist()[0]
result_2 = np.asarray(pd.read_csv('../result/misspelling_result_A.csv')).transpose().tolist()[0]

if __name__ == '__main__':
    result = []
    for i in range(50000):
        result.append(result_1[i] & result_2[i])
    with open('result/ccf_qianyan_qm_resultA.csv', 'w', encoding='utf8', newline='\n') as f:
        for row in result:
            f.write(str(row) + '\n')
