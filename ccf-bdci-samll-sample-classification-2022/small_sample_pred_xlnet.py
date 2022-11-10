from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
# from msilib.schema import Error
import os
import random
import sys
import datetime
from time import time, strftime, localtime

from tools import official_tokenization as tokenization, utils
import numpy as np
import torch
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modeling_nezha_small_sample import WEIGHTS_NAME, CONFIG_NAME
from optimization import BertAdam, warmup_linear
from transformers import BertConfig, AutoConfig, AutoTokenizer
from modeling import XLNetPrefixForSequenceClassification

logging.basicConfig(filename='./log/small-sample-predict-%s.log' % (strftime('%Y-%m-%d-%H-%M-%S', localtime(time()))),
                    encoding='utf8',
                    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None):
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


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

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

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None))
        return examples


def my_tokenize(text, tokenizer):
    tokens = text.split(" ")

    def is_in(word):
        if word not in tokenizer.vocab:
            return "<unk>"
        else:
            return word

    tokens_final = list(map(is_in, tokens))
    return tokens_final


def convert_examples_to_features(examples, max_seq_length, tokenizer, my_tokenization=False):
    """Loads a data file into a list of `InputBatch`s."""
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
            # Account for <cls>, <sep>, <sep> with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for <cls> and <sep> with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   <cls> is this jack ##son ##ville ? <sep> no it is not . <sep>
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   <cls> the dog is hairy . <sep>
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the <sep> token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to <cls>) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["<cls>"] + tokens_a + ["<sep>"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["<sep>"]
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

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        # logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
        # logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        # logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        # logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids))
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default='./data/',
                        type=str,
                        required=False,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--max_seq_length",
                        default=496,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--trained_model_dir", default='./trained_models/', type=str,
                        help="trained model for eval or predict")
    parser.add_argument("--output_dir",
                        default="./output/",
                        type=str,
                        required=False,
                        help="The output directory where the mompklodel predictions and checkpoints will be written.")
    parser.add_argument("--eval_batch_size",
                        default=1,
                        type=int,
                        help="Total batch size for eval.")
    args = parser.parse_args()

    processors = {
        "text-clf": TextClfProcessor
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    seed = 2022
    data_dir = args.data_dir
    output_dir = args.output_dir

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    if os.path.exists(output_dir) and os.listdir(output_dir):
        logger.info("dealing with existed files")
        # raise ValueError("Output directory ({}) already exists and is not empty.".format(output_dir))
        for file in os.listdir(output_dir):
            os.remove(output_dir + file)
        os.removedirs(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    task_name = 'text-clf'
    processor = processors[task_name]()

    trained_model_dir = args.trained_model_dir
    tokenizer = AutoTokenizer.from_pretrained(trained_model_dir)
    logger.info('vocab size is %d' % (len(tokenizer.vocab)))

    max_seq_length = args.max_seq_length
    eval_batch_size = args.eval_batch_size

    bert_config = AutoConfig.from_pretrained(trained_model_dir)
    model = XLNetPrefixForSequenceClassification(bert_config, num_labels=36)
    model.load_state_dict(torch.load(os.path.join(args.trained_model_dir, 'pytorch_model_0.bin')))
    logger.info('finish trained model loading!')

    # model = torch.load(os.path.join(trained_model_dir, "pytorch_model.bin"))

    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    eval_examples = processor.get_test_examples(data_dir)
    eval_features = convert_examples_to_features(eval_examples, max_seq_length, tokenizer)
    logger.info("***** Running testing *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

    model.eval()
    nb_eval_steps, nb_eval_examples = 0, 0
    id = 0
    probabilities = []
    results = []
    high_confidence_ids = []

    for input_ids, input_mask, segment_ids in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        with torch.no_grad():
            logits, prob = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

        logits = logits.detach().cpu().numpy()
        prob = prob.detach().cpu().numpy()
        for i in range(input_ids.size()[0]):
            probabilities.append(prob[i])
            prob_list = prob[i].tolist()
            results.append(np.argmax(prob_list))
            if max(prob_list) > 0.85:
                high_confidence_ids.append(id)
            id += 1

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1
    output_test_probabilities_file = os.path.join(output_dir, "test_probabilities.txt")
    output_test_results_file = os.path.join(output_dir, "test_results.txt")
    output_high_confidence_ids_file = os.path.join(output_dir, "high_confidence_ids.txt")
    with open(output_test_probabilities_file, "w", encoding='utf8') as writer:
        for prob in probabilities:
            writer.write(str(prob) + '\n')

    with open(output_test_results_file, "w", encoding='utf8') as writer:
        for result in results:
            writer.write(str(result) + '\n')

    with open(output_high_confidence_ids_file, "w", encoding='utf8') as writer:
        for id in high_confidence_ids:
            writer.write(str(id) + '\n')
    return probabilities, results


if __name__ == "__main__":
    prob, res = main()
