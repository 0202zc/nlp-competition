# 马上消费·天马杯 - 数字人播报资讯内容生成

@天马行空队

## 运行环境

`Ubuntu 22.04 LTS -> *.sh`

`Windows 10+ -> *.bat`

显卡：RTX A5000 24GB * 2

## 容器内目录结构

`/usr/local/msxf2022-nlp/`

```shell
├── data
│   ├── Test_A.csv
│   ├── Test_B.csv
│   ├── Train.csv
│   ├── testB_prompt.tsv
│   └── train_prompt.tsv
├── data_prompt.py
├── export_results.py
├── outputs
├── predict.py
├── pretrained_models
│   └── PromptCLUE-base
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── spiece.model
│       ├── spiece.vocab
│       └── tokenizer_config.json
├── run.sh
└── train.py
```

## 执行步骤

1. `sh build_docker.sh` (Ubuntu) 或 `build_docker.bat` (Windows)

2. 在运行的 Docker 容器中，执行 `sh run.sh` (Ubuntu) 或 `run.bat` (Windows)

> 预测结果文件为 `/usr/local/msxf2022-nlp/`下的 `result.xlsx`

