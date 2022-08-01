import codecs
import json
from collections import defaultdict


def get_kw2similar_words(fin, fout):
    kw2similar_words = defaultdict(set)  # 去重
    with codecs.open(fin, "r", "utf-8") as fr:
        for idx, line in enumerate(fr):
            try:
                row = line.strip().split(" ")
                if row[0][-1] == u"@":
                    continue
                for kw in row[1:]:
                    row.remove(kw)
                    kw_and_type = kw + row[0][-1]
                    kw2similar_words[kw_and_type].update(row[1:])
                    row.insert(-1, kw)
            except Exception as error:
                print("Error line", idx, line, error)
            if idx % 1000 == 0:
                print(idx)
    for kw, similar_words in kw2similar_words.items():
        kw2similar_words[kw] = list(similar_words)  # kw2similar_words = defaultdict(list)
    json.dump(kw2similar_words, open(fout, "w", encoding='utf-8'), ensure_ascii=False)


get_kw2similar_words(fin="./data/cilin.txt", fout="./data/kw2similar_words.json")

if __name__ == '__main__':
    kw2similar_words = json.load(open("./data/kw2similar_words.json", "r", encoding='utf-8'))
    keys_set = set(kw2similar_words.keys())
    kws = [u"群众", u"男人", u"女人", u"国王", u"皇后"]
    for kw in kws:
        for kw_and_type in [kw + u"=", kw + u"#"]:
            if kw_and_type in keys_set:
                print(kw_and_type, "/".join(kw2similar_words[kw_and_type]))
