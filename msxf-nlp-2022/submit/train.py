# 引入相应的包 Importing libraries
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import sys
import numpy as np
import pandas as pd
from rouge import Rouge
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import time, json
# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import get_linear_schedule_with_warmup

# rich: for a better display on terminal
from rich.table import Column, Table
from rich import box
from rich.console import Console

sys.setrecursionlimit(2000)

# 计算rouge用
rouge = Rouge()

# 做一些相关的配置(打印显示；GPU设置)
# define a rich console logger
console = Console(record=True)


# to display dataframe in ASCII format
def display_df(df):
    """display dataframe in ASCII format"""

    console = Console()
    table = Table(
        Column("source_text", justify="center"),
        Column("target_text", justify="center"),
        title="Sample Data",
        pad_edge=False,
        box=box.ASCII,
    )

    for i, row in enumerate(df.values.tolist()):
        table.add_row(row[0], row[1])

    # console.print(table) # TODO TODO TODO


# training logger to log training progress
training_logger = Table(
    Column("Epoch", justify="center"),
    Column("Steps", justify="center"),
    Column("Loss", justify="center"),
    title="Training Status",
    pad_edge=False,
    box=box.ASCII,
)

# Setting up the device for GPU usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SmallSampleDataSetClass(Dataset):
    """
    创建一个自定义的数据集，用于训练，必须包括两个字段：输入(如source_text)、输出（如target_text）
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, dataframe, tokenizer, source_len, target_len, source_text, target_text
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = self.data[target_text] if target_text is not None else None
        self.source_text = self.data[source_text]

    def __len__(self):
        """returns the length of dataframe"""
        return len(self.source_text)

    def __getitem__(self, index):
        
        if self.target_text is not None:
            """return the input ids, attention masks and target ids"""

            source_text = str(self.source_text[index])
            target_text = str(self.target_text[index])

            # cleaning data so as to ensure data is in string type
            source_text = " ".join(source_text.split())
            target_text = " ".join(target_text.split())

            source = self.tokenizer.batch_encode_plus(
                [source_text],
                max_length=self.source_len,
                pad_to_max_length=True,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            target = self.tokenizer.batch_encode_plus(
                [target_text],
                max_length=self.summ_len,
                pad_to_max_length=True,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            source_ids = source["input_ids"].squeeze()
            source_mask = source["attention_mask"].squeeze()
            target_ids = target["input_ids"].squeeze()
            target_mask = target["attention_mask"].squeeze()

            return {
                "source_ids": source_ids.to(dtype=torch.long),
                "source_mask": source_mask.to(dtype=torch.long),
                "target_ids": target_ids.to(dtype=torch.long),
                "target_ids_y": target_ids.to(dtype=torch.long),
            }
        else:
            """return the input ids, attention masks and target ids"""

            source_text = str(self.source_text[index])

            # cleaning data so as to ensure data is in string type
            source_text = " ".join(source_text.split())

            source = self.tokenizer.batch_encode_plus(
                [source_text],
                max_length=self.source_len,
                pad_to_max_length=True,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            
            source_ids = source["input_ids"].squeeze()
            source_mask = source["attention_mask"].squeeze()

            return {
                "source_ids": source_ids.to(dtype=torch.long),
                "source_mask": source_mask.to(dtype=torch.long)
            }


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def train(epoch, tokenizer, model, device, loader, optimizer, scheduler, ema):

    """
    用于训练的方法
    Function to be called for training with the parameters passed from main function

    """
    n_gpu = torch.cuda.device_count()

    model.train()
    time1 = time.time()
    for _, data in enumerate(loader, 0):
        y = data["target_ids"].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous() # target, from start to end(except end of token, <EOS>). e.g. "你好吗？"
        lm_labels = y[:, 1:].clone().detach() # target, for second to end.e.g."好吗？<EOS>"
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100 # releted to pad_token and loss. for detail, check here: https://github.com/Shivanandroy/T5-Finetuning-PyTorch/issues/3
        ids = data["source_ids"].to(device, dtype=torch.long) # input. e.g. "how are you?"
        mask = data["source_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )
        loss = outputs[0]
        if n_gpu > 1:
            loss = loss.mean()
            
        # 每100步打印日志
        if _ % 100 == 0 and _ != 0 or _ == len(loader) - 1:
            time2 = time.time()
            print("Step:", _,"epoch: " + str(epoch) + "; loss:{:.4f}; each step's time spent:{:.2f}".format(loss.detach().cpu().numpy(), float(time2-time1) / float(_ + 0.0001)))
            # training_logger.add_row(str(epoch), str(_), str(loss))
            # console.print(training_logger)
            # console.log(f"Step: {_}, epoch: {epoch}; loss: {loss.detach().cpu().numpy()}; each step's time spent:{float(time2-time1) / float(_ + 0.0001)}\n")
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update()
        scheduler.step()


metric_keys = ['main', 'rouge-1', 'rouge-2', 'rouge-l']

def compute_rouge(source, target, unit='word'):
    """计算rouge-1、rouge-2、rouge-l
    """
    # if unit == 'word':
    #     source = jieba.cut(source, HMM=False)
    #     target = jieba.cut(target, HMM=False)
    source, target = ' '.join(source), ' '.join(target)
    try:
        scores = rouge.get_scores(hyps=source, refs=target)
        return {
            'rouge-1': scores[0]['rouge-1']['f'],
            'rouge-2': scores[0]['rouge-2']['f'],
            'rouge-l': scores[0]['rouge-l']['f'],
        }
    except ValueError:
        return {
            'rouge-1': 0.0,
            'rouge-2': 0.0,
            'rouge-l': 0.0,
        }

    
def compute_metrics(source, target, unit='word'):
    """计算所有metrics
    """
    metrics = compute_rouge(source, target, unit)
    metrics['main'] = (
            metrics['rouge-1'] * 0.2 + metrics['rouge-2'] * 0.4 +
            metrics['rouge-l'] * 0.4
    )
    return metrics


def validate(epoch, tokenizer, model, device, loader, max_length, ema):

    """
    用于验证的方法：输入用于验证的数据，返回模型预测的结果和正确的标签
    Function to evaluate model for predictions

    """
    model.eval()
    ema.apply_shadow()
    predictions = []
    actuals = []
    total_metrics = {k: 0.0 for k in metric_keys}
    count = 0
    
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
              input_ids = ids,
              attention_mask = mask, 
              max_length=max_length, 
              num_beams=2,
              repetition_penalty=2.5, 
              length_penalty=1.0, 
              early_stopping=True
              )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
            
            for i in range(len(preds)):
                metrics = compute_metrics(preds[i], target[i])
                for k, v in metrics.items():
                    total_metrics[k] += v
            count += len(preds)
                
            if _ % 10 == 0:
                console.print(f'Completed {_}')
                print("preds: %s\ntarget: %s" % (preds[0], target[0]))

            predictions.extend(preds)
            actuals.extend(target)
            
    avg_metrics = {k: v / count for k, v in total_metrics.items()}
                
    print(avg_metrics)
    console.log(f"{avg_metrics}\n")
    ema.restore()
            
    return predictions, actuals


# 训练类：整合数据集类、训练方法、验证方法，加载数据进行训练并验证训练过程的效果
def T5Trainer(
    dataframe, source_text, target_text, model_params, output_dir="./outputs/", train_mode=True, val_mode=True
):
    """
    T5 trainer
    """
    n_gpu = torch.cuda.device_count()
    
    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(model_params["SEED"])  # pytorch random seed
    np.random.seed(model_params["SEED"])  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # logging
    console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])

    # Defining the model. We are using PromptCLUE model and added a Language model layer on top for generation of prediction.
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    # model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"], max_seq_len=model_params["MAX_SOURCE_TEXT_LENGTH"])
    model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
    model = model.to(device)
    
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
        # model = model.module.to(device)

    # logging
    console.log(f"[Data]: Reading data...\n")

    # Importing the raw dataset
    dataframe = dataframe[[source_text, target_text]]
    # display_df(dataframe.head(2))

    # Creation of Dataset and Dataloader
    # Defining the train size So 99.98% of the data will be used for training and the rest for validation.
    train_size = 0.9998
    train_dataset = dataframe.sample(frac=train_size, random_state=model_params["SEED"])
    val_dataset = dataframe.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    # 打印数据集相关日志：数据量、训练步数
    console.print(f"FULL Dataset: {dataframe.shape}")
    console.print(f"TRAIN Dataset: {train_dataset.shape}")
    console.print(f"VALID Dataset: {val_dataset.shape}\n")
    total_train_steps = int((train_dataset.shape[0] * model_params["TRAIN_EPOCHS"]) / model_params["TRAIN_BATCH_SIZE"])
    console.print(f"Total Train Steps: {total_train_steps}\n")

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = SmallSampleDataSetClass(
        train_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )
    val_set = SmallSampleDataSetClass(
        val_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )
    # Defining the parameters for creation of dataloaders
    train_params = {
        "batch_size": model_params["TRAIN_BATCH_SIZE"],
        "shuffle": True,
        "num_workers": 4,
    }

    val_params = {
        "batch_size": model_params["VALID_BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 4,
    }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    # optimizer = torch.optim.Adam(
    #     params=model.parameters(), lr=model_params["LEARNING_RATE"]
    # )
    optimizer = torch.optim.AdamW(model.parameters(), model_params["LEARNING_RATE"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_train_steps, num_training_steps=total_train_steps)
    
    ema = EMA(model, 0.992)
    ema.register()

    # Training loop
    console.log(f"[Initiating Fine Tuning]...\n")

    for epoch in range(model_params["TRAIN_EPOCHS"]):
        if train_mode:
            # 1) train for one epoch
            train(epoch, tokenizer, model, device, training_loader, optimizer, scheduler, ema)

            # 2) save model for each epoch
            console.log(f"[Saving Model]...\n")
            path = os.path.join(output_dir, "model_files")
            model.module.save_pretrained(path)
            tokenizer.save_pretrained(path)

        torch.cuda.empty_cache()

        if val_mode and (epoch == 0 or epoch == model_params["TRAIN_EPOCHS"] - 1):
            # 3) evaluating test dataset
            console.log(f"[Initiating Validation]...\n")
            with torch.no_grad(): # add 2022.10.4
                #for epoch in range(model_params["VAL_EPOCHS"]):
                predictions, actuals = validate(epoch, tokenizer, model.module, device, val_loader, model_params["MAX_TARGET_TEXT_LENGTH"], ema)
                final_df = pd.DataFrame({"Generated Text": predictions, "Actual Text": actuals})
                final_df.to_csv(os.path.join(output_dir, "predictions.csv"), encoding='utf8', index=None, sep=',')

    console.save_text(os.path.join(output_dir, "logs.txt"))

    console.log(f"[Validation Completed.]\n")
    console.print(
        f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n"""
    )
    console.print(
        f"""[Validation] Generation on Validation data saved @ {os.path.join(output_dir,'predictions.csv')}\n"""
    )
    console.print(f"""[Logs] Logs saved @ {os.path.join(output_dir,'logs.txt')}\n""")


if __name__ == '__main__':
    # 定义模型的参数 let's define model parameters specific to T5
    model_params = {
        "MODEL": "pretrained_models/PromptCLUE-base",
        # model_type pretrained_models/PromptCLUE-base & outputs/prompt/model_files/
        "TRAIN_BATCH_SIZE": 12,  # training batch size, 8
        "VALID_BATCH_SIZE": 16,  # validation batch size, 8
        "TRAIN_EPOCHS": 4,  # number of training epochs
        "VAL_EPOCHS": 1,  # number of validation epochs
        "LEARNING_RATE": 2e-4,  # learning rate
        "MAX_SOURCE_TEXT_LENGTH": 500,  # max length of source text, 512
        "MAX_TARGET_TEXT_LENGTH": 420,  # max length of target text,64
        "SEED": 2022,  # set seed for reproducibility
    }

    # 训练模型
    # dataframe必须有2列:
    #   - input: 文本输入
    #   - target: 目标输出
    df = pd.read_csv('data/train_prompt.tsv', sep='\t', encoding='utf8')  # 数据量：1200k数据。
    print("df.head:", df.head(n=5))
    print("df.shape:", df.shape)
    # 显存占用说明：如果运行现在显存不足，请使用nvidia-smi查看显存；如果显卡多数被占用了，请重启colab程序
    T5Trainer(
        dataframe=df,
        source_text="input",
        target_text="target",
        model_params=model_params,
        output_dir="outputs/",
        train_mode=True,
        val_mode=True
    )

    print("Model training has finished!")

    torch.cuda.empty_cache()
