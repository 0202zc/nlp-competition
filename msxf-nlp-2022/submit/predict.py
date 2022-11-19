# 引入相应的包 Importing libraries
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import time, json
# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
# from models.modeling_t5 import T5ForConditionalGeneration

# rich: for a better display on terminal
from rich.table import Column, Table
from rich import box
from rich.console import Console

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


def testing(tokenizer, model, device, loader, max_length):

    """
    用于预测的方法：输入用于预测的数据，返回模型预测的结果
    Function for predictions

    """
    
    model.eval()
    predictions = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
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
            
            print(_, preds[0])
            if _ % 10 == 0:
                console.print(f'Completed {_}')

            predictions.extend(preds)
    return predictions


# 训练类：整合数据集类、训练方法、验证方法，加载数据进行训练并验证训练过程的效果
def T5Tester(
    dataframe, source_text, model_params, output_dir="./outputs/"
):
    """
    T5 tester
    """
    
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
    model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
    model = model.to(device)

    # logging
    console.log(f"[Data]: Reading data...\n")

    # Importing the raw dataset
    # dataframe = dataframe[source_text]
    # display_df(dataframe.head(2))

    # Creation of Dataset and Dataloader
    # Defining the train size So 94% of the data will be used for training and the rest for validation.
    test_dataset = dataframe
    
    # 打印数据集相关日志：数据量、训练步数
    console.print(f"FULL Dataset: {dataframe.shape}")
    console.print(f"TEST Dataset: {test_dataset.shape}")

    # Creating the Training and Validation dataset for further creation of Dataloader
    testing_set = SmallSampleDataSetClass(
        test_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text=source_text,
        target_text=None
    )

    # Defining the parameters for creation of dataloaders
    test_params = {
        "batch_size": model_params["TEST_BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 4
    }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    testing_loader = DataLoader(testing_set, **test_params)
    
    # 3) evaluating test dataset
    console.log(f"[Initiating Prediction]...\n")
    with torch.no_grad(): # add 2022.10.4
        #for epoch in range(model_params["VAL_EPOCHS"]):
        predictions = testing(tokenizer, model, device, testing_loader, model_params["MAX_TARGET_TEXT_LENGTH"])
        final_df = pd.DataFrame({"Generated Text": predictions})
        final_df.to_csv(os.path.join(output_dir, "predictions.csv"), encoding='utf8', index=None)

    console.save_text(os.path.join(output_dir, "logs.txt"))

    console.log(f"[Prediction Completed.]\n")
    console.print(
        f"""[Prediction] Generation on Testing data saved @ {os.path.join(output_dir,'predictions.csv')}\n"""
    )
    console.print(f"""[Logs] Logs saved @ {os.path.join(output_dir,'logs.txt')}\n""")


if __name__ == '__main__':
    # 定义模型的参数 let's define model parameters specific to T5
    model_params = {
        "MODEL": "outputs/model_files/",  # model_type
        "TEST_BATCH_SIZE": 50,  # training batch size
        "MAX_SOURCE_TEXT_LENGTH": 330,  # max length of source text
        "MAX_TARGET_TEXT_LENGTH": 420,  # max length of target text
        "SEED": 2022,  # set seed for reproducibility
    }

    # 预测结果
    # dataframe必须有2列:
    #   - input: 文本输入
    #   - target: 目标输出
    df = pd.read_csv('data/testB_prompt.tsv', sep='\t', encoding='utf8', header=0, names=["input"])  # 数据量：1675数据。
    print("df.head:",df.head(n=5))
    print("df.shape:",df.shape)
    # 显存占用说明：如果运行现在显存不足，请使用nvidia-smi查看显存；如果显卡多数被占用了，请重启colab程序
    T5Tester(
        dataframe=df,
        source_text="input",
        model_params=model_params,
        output_dir="outputs/prediction/",
    )
    torch.cuda.empty_cache() 
