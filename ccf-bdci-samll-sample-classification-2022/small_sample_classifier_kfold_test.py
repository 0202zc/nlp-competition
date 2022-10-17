# coding=utf-8
# 2020.12.18-Changed for fine-tuning NEZHA model.
# Huawei Technologies Co., Ltd. <foss@huawei.com> 
# Copyright 2020 Huawei Technologies Co., Ltd.
# Copyright 2018 The Google AI Language Team Authors, The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import datetime
from time import time, strftime, localtime

# from FGM import FGM
from tools import official_tokenization as tokenization, utils
import numpy as np
import pandas as pd
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modeling_nezha_small_sample import WEIGHTS_NAME, CONFIG_NAME
from modeling import SmallSampleClassification
from optimization import BertAdam, warmup_linear
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection._split import StratifiedKFold
from transformers import AutoTokenizer

# from transformers import NezhaModel, BertTokenizer

logging.basicConfig(filename='./log/small-sample-train-%s.log' % (strftime('%Y-%m-%d-%H-%M-%S', localtime(time()))),
                    encoding='utf8',
                    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


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


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = unicode(line, 'utf-8')
                lines.append(line)
            return lines


class TextClfProcessor(DataProcessor):
    """Processor for the TextClf. label tab text version"""

    def __init__(self):
        self.label_list = []

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return self.label_list

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        if set_type == 'test':
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                guid = "%s-%s" % (set_type, i)
                text_a = line[0]
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=None))
        else:
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                guid = "%s-%s" % (set_type, i)
                text_a = line[1]
                label = line[0]
                if set_type == 'train' and (label not in self.label_list):
                    self.label_list.append(label)
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


def my_tokenize(text, tokenizer):
    tokens = text.split(" ")

    def is_in(word):
        if word not in tokenizer.vocab:
            return "[UNK]"
        else:
            return word

    tokens_final = list(map(is_in, tokens))
    return tokens_final


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, label_map_training=None,
                                 my_tokenization=False):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {}
    if label_list:
        label_map = {label: i for i, label in enumerate(label_list)}
    elif label_map_training:
        label_map = label_map_training
    features = []
    for (ex_index, example) in enumerate(examples):
        if my_tokenization:
            tokens_a = my_tokenize(example.text_a, tokenizer)
        else:
            tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            if False:  # " " in example.text_b:
                tokens_b = my_tokenize(example.text_b, tokenizer)
            else:
                tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        if example.label == None:
            label_id = 9999  # padding
        else:
            label_id = label_map[example.label]
        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        # logger.info("tokens: %s" % " ".join(
        #     [str(x) for x in tokens]))
        # logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        # logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        # logger.info(
        #     "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        # logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def my_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def accuracy(out, labels, topk=1):
    batch_results = []
    corr_count = 0
    batch_size = out.shape[0]
    for i in range(batch_size):
        tmp_out = out[i]
        probs = my_softmax(tmp_out)
        max_prob_indices = np.argpartition(tmp_out, -topk)[-topk:].tolist()
        if labels[i] in max_prob_indices:
            corr_count += 1
        batch_results.append([labels[i], max_prob_indices[-1]])
    return corr_count, batch_results


def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=False,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")

    # trained_model_file
    parser.add_argument("--trained_model_dir", default=None, type=str,
                        help="trained model for eval or predict")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--my_tokenization",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    processors = {
        "text-clf": TextClfProcessor
    }

    num_labels_task = {
        "text-clf": 0
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval and not args.do_test:
        # raise ValueError("At least one of `do_train` or `do_eval(test)` must be True.")
        for file in os.listdir(args.output_dir):
            os.remove(args.output_dir + file)
        os.removedirs(args.output_dir)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        # raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
        for file in os.listdir(args.output_dir):
            os.remove(args.output_dir + file)
        os.removedirs(args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    num_labels = num_labels_task[task_name]

    label_list = processor.get_labels()
    if args.bert_model:
        # tokenizer = tokenization.BertTokenizer(vocab_file=os.path.join(args.bert_model, 'vocab.txt'), do_lower_case=False)
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    elif args.trained_model_dir:
        # tokenization.BertTokenizer(vocab_file=os.path.join(args.trained_model_dir, 'vocab.txt'), do_lower_case=False)
        tokenizer = AutoTokenizer.from_pretrained(args.trained_model_dir)
    logger.info('vocab size is %d' % (len(tokenizer.vocab)))
    label_map_reverse = {}
    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        if task_name == 'text-clf':
            num_labels = len(label_list)
        label_map = {label: i for i, label in enumerate(label_list)}
        label_file = os.path.join(args.output_dir, "label_map_training.txt")
        with open(label_file, "w") as writer:
            for (k, v) in label_map.items():
                writer.write(str(k))
                writer.write('\t')
                writer.write(str(v))
                writer.write('\n')
        label_map_reverse = {v: k for k, v in label_map.items()}
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
    else:
        train_examples = processor.get_train_examples(args.data_dir)
        label_map = {label: i for i, label in enumerate(label_list)}
        label_map_reverse = {v: k for k, v in label_map.items()}
        num_labels = len(label_list)
    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                   'distributed_{}'.format(args.local_rank))

    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, my_tokenization=args.my_tokenization)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        test_examples = processor.get_test_examples(args.data_dir)
        test_features = convert_examples_to_features(test_examples, None, args.max_seq_length, tokenizer)
        test_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        test_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        test_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)

        # 初始化
        # fgm = FGM(model)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

        for fold, (train_index, valid_index) in enumerate(skf.split(all_label_ids, all_label_ids)):
            # if fold < 0:
            #     continue
            logger.info('================     fold {}        ==============='.format(fold))
            # 处理模型输入数据
            train_input_ids = torch.as_tensor(all_input_ids[train_index], dtype=torch.long)
            train_input_mask = torch.as_tensor(all_input_mask[train_index], dtype=torch.long)
            train_segment_ids = torch.as_tensor(all_segment_ids[train_index], dtype=torch.long)
            train_label = torch.as_tensor(all_label_ids[train_index], dtype=torch.long)

            valid_input_ids = torch.as_tensor(all_input_ids[valid_index], dtype=torch.long)
            valid_input_mask = torch.as_tensor(all_input_mask[valid_index], dtype=torch.long)
            valid_segment_ids = torch.as_tensor(all_segment_ids[valid_index], dtype=torch.long)
            valid_label = torch.as_tensor(all_label_ids[valid_index], dtype=torch.long)

            train = TensorDataset(train_input_ids, train_input_mask, train_segment_ids, train_label)
            valid = TensorDataset(valid_input_ids, valid_input_mask, valid_segment_ids, valid_label)
            test = TensorDataset(test_input_ids, test_input_mask, test_segment_ids)

            train_loader = DataLoader(train, batch_size=args.train_batch_size, shuffle=True)
            valid_loader = DataLoader(valid, batch_size=args.train_batch_size, shuffle=False)
            test_loader = DataLoader(test, batch_size=128, shuffle=False)

            num_train_optimization_steps = int(
                len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
            logger.info("  Num steps = %d", num_train_optimization_steps)

            if args.trained_model_dir:
                # config = BertConfig(os.path.join(args.trained_model_dir, 'bert_config.json'))
                # model = BertForSequenceClassification(config, num_labels=num_labels)
                # model.load_state_dict(torch.load(os.path.join(args.trained_model_dir, 'pytorch_model.bin')))
                model = SmallSampleClassification(args.trained_model_dir, max_seq_len=args.max_seq_length, num_labels=num_labels)
                logger.info('finish trained model loading!')
            elif args.bert_model:
                print('init model...')
                # bert_config = BertConfig.from_json_file(os.path.join(args.bert_model, 'bert_config.json'))
                # model = BertForSequenceClassification(bert_config, num_labels=num_labels)
                # utils.torch_show_all_params(model)

                model = SmallSampleClassification(args.bert_model, max_seq_len=args.max_seq_length, num_labels=num_labels)
                utils.torch_init_model(model, os.path.join(args.bert_model, 'pytorch_model.bin'), prefix='nezha')

            if args.fp16:
                model.half()

            model.to(device)
            if args.local_rank != -1:
                try:
                    from apex.parallel import DistributedDataParallel as DDP
                except ImportError:
                    raise ImportError(
                        "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

                model = DDP(model)
            elif n_gpu > 1:
                model = torch.nn.DataParallel(model)

            # Prepare optimizer
            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            if args.fp16:
                try:
                    from apex.optimizers import FP16_Optimizer
                    from apex.optimizers import FusedAdam
                except ImportError:
                    raise ImportError(
                        "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

                optimizer = FusedAdam(optimizer_grouped_parameters,
                                      lr=args.learning_rate,
                                      bias_correction=False,
                                      max_grad_norm=1.0)
                if args.loss_scale == 0:
                    optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
                else:
                    optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

            else:
                optimizer = BertAdam(optimizer_grouped_parameters,
                                     lr=args.learning_rate,
                                     warmup=args.warmup_proportion,
                                     t_total=num_train_optimization_steps)

            total_steps = int(args.num_train_epochs) * len(train_loader)
            # ema = EMA(model, 0.999)
            # ema.register()

            num_epoch = 0
            global_step = 0
            nb_tr_steps = 0
            tr_loss = 0

            for _ in trange(int(args.num_train_epochs), desc="Epoch"):
                model.train()
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0

                # optimizer.zero_grad(set_to_none=True)
                # scaler = GradScaler()

                for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                    batch = tuple(t.to(device, non_blocking=True) for t in batch)
                    input_ids, input_mask, segment_ids, label_ids = batch

                    # 正常训练
                    loss = model(input_ids, segment_ids, input_mask, label_ids)

                    if n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu.
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    if args.fp16:
                        optimizer.backward(loss)
                    else:
                        loss.backward()

                    # 对抗训练
                    # fgm.attack()  # 修改embedding
                    # # optimizer.zero_grad() # 梯度累加，不累加去掉注释
                    # loss_sum = model(input_ids, segment_ids, input_mask, label_ids)  # 累加对抗训练的梯度
                    # loss_sum.backward()
                    # fgm.restore()  # 恢复Embedding的参数

                    # ema.update()

                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        if args.fp16:
                            # modify learning rate with special warm up BERT uses
                            # if args.fp16 is False, BertAdam is used that handles this automatically
                            lr_this_step = args.learning_rate * warmup_linear(
                                global_step / num_train_optimization_steps,
                                args.warmup_proportion)
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = lr_this_step
                        optimizer.step()
                        optimizer.zero_grad()
                        # scaler.step(optimizer)
                        # scaler.update()
                        global_step += 1
                num_epoch += 1

                ## begin to evaluate
                eval_all_result = []

                # eval_examples = processor.get_dev_examples(args.data_dir)
                # eval_features = convert_examples_to_features(
                #     eval_examples, label_list, args.max_seq_length, tokenizer, my_tokenization=args.my_tokenization)
                logger.info("***** Running  %d -th evaluation *****" % num_epoch)
                logger.info("  Batch size = %d", args.eval_batch_size)

                # ema.apply_shadow()
                model.eval()
                eval_loss, eval_accuracy = 0, 0
                nb_eval_steps, nb_eval_examples = 0, 0
                results = []
                best_f1 = 0.

                for input_ids, input_mask, segment_ids, label_ids in tqdm(valid_loader, desc="Evaluating"):
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_ids = label_ids.to(device)

                    with torch.no_grad():
                        # with autocast():
                        tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                        logits, prob = model(input_ids, segment_ids, input_mask)

                    logits = logits.detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    prob = prob.detach().cpu().numpy()
                    tmp_eval_accuracy, batch_result = accuracy(logits, label_ids)
                    for i in range(input_ids.size()[0]):
                        eval_all_result.append(batch_result[i])
                        prob_list = prob[i].tolist()
                        results.append(np.argmax(prob_list))

                    eval_loss += tmp_eval_loss.mean().item()
                    eval_accuracy += tmp_eval_accuracy

                    nb_eval_examples += input_ids.size(0)
                    nb_eval_steps += 1

                eval_loss = eval_loss / nb_eval_steps
                eval_accuracy = eval_accuracy / nb_eval_examples
                loss = tr_loss / nb_tr_steps if args.do_train else None

                eval_precision = precision_score(y_true=all_label_ids[valid_index], y_pred=results, average='macro',
                                                 zero_division='warn')
                eval_recall = recall_score(y_true=all_label_ids[valid_index], y_pred=results, zero_division='warn',
                                           average='macro')
                eval_f1 = f1_score(y_true=all_label_ids[valid_index], y_pred=results, average='macro')
                result = {'epoch': num_epoch,
                          'eval_loss': eval_loss,
                          'eval_accuracy': eval_accuracy,
                          'global_step': global_step,
                          'loss': loss,
                          'eval_precision': eval_precision,
                          'eval_recall': eval_recall,
                          'eval_macro_f1': eval_f1}

                logger.info(result)
                output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                epoch_eval_result_file = os.path.join(args.output_dir, str(num_epoch) + "th_epoch_eval_results.txt")
                with open(output_eval_file, "a") as writer:
                    logger.info("***** %d th epoch eval results *****" % num_epoch)
                    writer.write("***%d th epoch result***" % num_epoch)
                    for key in sorted(result.keys()):
                        logger.info("  %s = %s", key, str(result[key]))
                        writer.write("%s = %s\n" % (key, str(result[key])))
                if num_epoch < 0:
                    continue
                with open(epoch_eval_result_file, "w") as writer:
                    for element in eval_all_result:
                        tokens_sample = 'Text'
                        result_sample = element
                        writer.write(str(tokens_sample))
                        writer.write('\t')
                        for ele in result_sample:
                            writer.write(label_map_reverse[ele])
                            writer.write('\t')
                        writer.write('\n')

                # Save a trained model and the associated configuration
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                if best_f1 < eval_f1:
                    best_f1 = eval_f1
                    output_model_file = os.path.join(args.output_dir,
                                                     WEIGHTS_NAME.replace(".pth", "_") + "{}.bin".format(fold))
                    torch.save(model_to_save.state_dict(), output_model_file)
                    torch.save(model_to_save, "./trained_models/pytorch_model_{}.bin".format(fold))

                output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
                with open(output_config_file, 'w') as f:
                    f.write(model_to_save.config.to_json_string())

                torch.cuda.empty_cache()

                if num_epoch == int(args.num_train_epochs):
                    break

            num_epoch = 0
            global_step = 0
            nb_tr_steps = 0
            tr_loss = 0

            # 重载模型字典
            state_dict = {}
            origin_state_dict = torch.load(
                args.output_dir + WEIGHTS_NAME.replace(".pth", "_") + "{}.bin".format(fold))
            for key in origin_state_dict:
                val = origin_state_dict[key]
                new_key = key.replace('module.', "")
                state_dict[new_key] = val
            state_dict = origin_state_dict
            model.load_state_dict(state_dict, strict=False)
            torch.cuda.empty_cache()

            # model.load_state_dict(torch.load(args.output_dir + WEIGHTS_NAME.replace(".pth", "_") + "{}_{}.bin".format(fold, num_epoch)))
            model.eval()
            nb_eval_steps, nb_eval_examples = 0, 0
            id = 1
            results = []
            for input_ids, input_mask, segment_ids in tqdm(test_loader, desc="Evaluating"):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)

                with torch.no_grad():
                    logits, prob = model(input_ids, segment_ids, input_mask)

                logits = logits.detach().cpu().numpy()
                prob = prob.detach().cpu().numpy()
                for i in range(input_ids.size()[0]):
                    prob_list = prob[i].tolist()
                    results.append(np.argmax(prob_list))
                    id += 1

                nb_eval_examples += input_ids.size(0)
                nb_eval_steps += 1
            torch.cuda.empty_cache()
            output_test_results_file = os.path.join(args.output_dir, "test_results_{}.txt".format(fold))
            with open(output_test_results_file, "w", encoding='utf8') as writer:
                for result in results:
                    writer.write(str(result) + '\n')

            num_epoch = 0
            global_step = 0
            nb_tr_steps = 0
            tr_loss = 0


if __name__ == "__main__":
    main()