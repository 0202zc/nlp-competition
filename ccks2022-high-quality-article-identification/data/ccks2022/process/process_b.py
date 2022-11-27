import json
import pandas as pd
import numpy as np


def process_data_set():
    train_df = pd.read_json('train.json', encoding='utf8', lines=True)
    test_df = pd.read_json('test.unlabel.json', encoding='utf8', lines=True)
    # ----------------特征工程----------------
    # 将Topic(Label)编码
    train_df['label'], lbl = pd.factorize(train_df['label'])

    # 将论文的标题与摘要组合为 text 特征
    train_df['title'] = train_df['title'].apply(lambda x: x.strip())
    train_df['content'] = train_df['content'].fillna('').apply(lambda x: x.strip())
    train_df['text'] = train_df['title'] + '[SEP]' + train_df['content']
    train_df['text'] = train_df['text'].str.lower()

    test_df['title'] = test_df['title'].apply(lambda x: x.strip())
    test_df['content'] = test_df['content'].fillna('').apply(lambda x: x.strip())
    test_df['text'] = test_df['title'] + '[SEP]' + test_df['content']
    test_df['text'] = test_df['text'].str.lower()

    for i in range(train_df.shape[0]):
        train_df['text'].iloc[i] = train_df['text'].iloc[i].replace('  ', '').replace('\n', ' ').replace('<br/>',
                                                                                                         ' ').replace(
            "《", " ").replace("》", " ").replace("？", " ").replace("【", "").replace("】", "")
    train_df.to_csv('train.tsv', encoding='utf8', sep='\t', index=None)

    for i in range(test_df.shape[0]):
        test_df['text'].iloc[i] = test_df['text'].iloc[i].replace('  ', '').replace('\n', ' ').replace('<br/>',
                                                                                                       ' ').replace("《",
                                                                                                                    " ").replace(
            "》", " ").replace("？", " ").replace("【", "").replace("】", "")
    train_df.to_csv('train.tsv', encoding='utf8', sep='\t', index=None)

    train_out = pd.concat(
        [train_df.drop(['url', 'title', 'pub_time', 'content', 'entities'], axis=1), train_df.iloc[:, -3:-2]], axis=1)
    train_out.to_csv('result/train.tsv', encoding='utf8', sep='\t', index=None)

    test_out = pd.concat(
        [test_df.drop(['url', 'title', 'pub_time', 'content', 'entities'], axis=1), test_df.iloc[:, -2:-1]], axis=1)
    test_out.to_csv('result/test.tsv', encoding='utf8', sep='\t', index=None)


def process_json_list():
    df = pd.read_json('test.unlabel.json', encoding='utf8', lines=True)
    labels = pd.read_csv("test_results.txt", encoding='utf8', header=None, names=['labels'])
    s = ""
    for i in range(df.shape[0]):
        dic = dict()
        dic['url'] = df.iloc[i, :]["url"]
        dic['label'] = int(labels.iloc[i, :]["labels"])
        s += json.dumps(dic, ensure_ascii=False) + "\n"
        if i == 0:
            print(s[:100])
    with open("result.txt", encoding="utf-8", mode='w') as f:
        f.write(s)
        f.close()


if __name__ == '__main__':
    process_data_set()
