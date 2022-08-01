# 2021-CCF-BDCI-千言-问题匹配鲁棒性评测

### 介绍
基于规则的训练集生成及数据修改项目

### 目录文件备注

| 路径                                   | 备注                          |  状态  |
|--------------------------------------|-----------------------------| ----  |
| corrector                            | 错别字纠正                       | 在用 <br>（/pinyin_test.py） |
| dataset                              | 数据集（单项，已合并 train, dev, test） | 在用 |
| kfold_csv/                           | 加权融合                        | 在用 |
| syntactic-structure/                 | 特定语句数据生成                    | 最近不用 |
| qianyan-qm-pytorch/                  | Pytorch 版本的 baseline        | 在用 |
| qianyan-qm-paddlepaddle/             | 飞桨版本的 baseline              | 在用 |
| submit_result_A/                     | A榜结果文件（文件名即为分数）             | 在用 |
| similarity/                          | nlpcda 同、近义词替换              | 不适用 |
| synonyms-work/                       | 同义词测试                       | 不适用 |
| translate/                           | 回译                          | 不适用 |
| corrector/result/single-part-result/ | 单项训练集A榜结果文件                 | 在用 |
| corrector/pinyin_test.py             | 同音字纠正 + 输出结果文件（新）           | 在用 |
| transform_test_A.py                  | 同音字替换 + 输出结果文件（旧）           | 最近不用 |
| ccf-qianyan-test-score.xlsx          | A榜测试结果记录表                   | 在用 |
| merge_result_A.py                    | 合并A榜测试集和预测标签                | 在用 |
