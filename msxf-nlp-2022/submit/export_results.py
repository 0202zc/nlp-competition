import pandas as pd

testB_df = pd.read_csv('data/Test_B.csv', encoding='utf8', header=0)
news = pd.read_csv('outputs/prediction/predictions.csv', encoding='utf8', header=0, names=['News'], sep='\t')
testB_df['News'] = news['News']
testB_df[['ID', 'News']].to_excel('result.xlsx', encoding='utf8', index=False)

print('Result file has been exported: result.xlsx')