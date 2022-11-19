import pandas as pd

if __name__ == '__main__':
    train_df = pd.read_csv('data/Train.csv', encoding='utf8', header=0)
    testB_df = pd.read_csv('data/Test_B.csv', encoding='utf8', header=0)

    train_elements_df = train_df['Elements'].str.replace('｜', '|', regex=False).str.split('[SEP]', regex=False, expand=True)
    train_elements_df.columns = ['title', 'keywords']
    train_elements_df['keywords'] = train_elements_df['keywords'].str.replace('#', ';')

    train_elements_df['input'] = '根据标题和关键词生成文章：_标题：' + train_elements_df['title'].str.strip() + '_关键词：' + train_elements_df['keywords'].str.strip() + '_答案：'
    train_elements_df['target'] = train_df['News'].str.replace('｜', '|', regex=False).str.replace('', '', regex=False).str.strip()

    train_elements_df[['input', 'target']].to_csv('data/train_prompt.tsv', sep='\t', encoding='utf8', index=False)

    testB_elements_df = testB_df['Elements'].str.replace('｜', '|', regex=False).str.split('[SEP]', regex=False, expand=True)
    testB_elements_df.columns = ['title', 'keywords']
    testB_elements_df['keywords'] = testB_elements_df['keywords'].str.replace('#', ';').str.replace('​', "", regex=False)

    testB_elements_df['input'] = '根据标题和关键词生成文章：_标题：' + testB_elements_df['title'].str.strip() + '_关键词：' + testB_elements_df['keywords'].str.strip() + '_答案：'

    testB_elements_df['input'].to_csv('data/testB_prompt.tsv', sep='\t', encoding='utf8', index=False)

    print('Data preprocssing has finished!')