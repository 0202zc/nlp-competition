import jieba
import jieba.analyse
import pypinyin
from Pinyin2Hanzi import DefaultDagParams
from Pinyin2Hanzi import dag


def pinyin_2_hanzi(pinyin_list):
    dag_params = DefaultDagParams()
    result = dag(dag_params, pinyin_list, path_num=10, log=True)
    res = []
    for item in result:
        # score = item.score
        res.append(item.path)
    print(res)
    return res


def cut_sentence(sentence):
    seg = jieba.lcut(sentence, cut_all=True)
    # keywords = jieba.analyse.extract_tags(sentence, topK=5, withWeight=True)
    # print(keywords)
    return seg


# template 为句子模板
def generate(template):
    pinyin_list = [pypinyin.pinyin(w, heteronym=True, style=pypinyin.STYLE_NORMAL)[0][0] for w in template]

    results = pinyin_2_hanzi(pinyin_list)

    sentences = []
    for result in results:
        temp = ''
        for i in range(len(result)):
            temp += result[i]
        if temp != template:
            sentences.append(temp)

    output = []
    for sentence in sentences:
        output.append('{}\t{}\t1'.format(template, sentence))

    return output


if __name__ == '__main__':
    output = generate('不知道密码再解锁')

    for o in output:
        print(o)
