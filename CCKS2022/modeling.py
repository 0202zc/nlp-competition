import torch
import modeling_nezha_ccks2022

from torch import nn
from focal_loss import FocalLoss
from modeling_nezha_ccks2022 import BertPreTrainedModel, BertModel
from transformer import Transformer, TransformerForPretraining


class NezhaForPretrain(BertPreTrainedModel):
    def __init__(self, bert_config):
        super(NezhaForPretrain, self).__init__(bert_config)
        self.config = bert_config
        self.dropout = nn.Dropout(0.2)
        self.bert = BertModel(self.config)
        self.apply(self.init_bert_weights)
        self.gru = nn.GRU(self.config.hidden_size, self.config.hidden_size, num_layers=2, bidirectional=True)
        self.transformer = TransformerForPretraining(max_len=512, hidden_size=self.config.hidden_size)
        self.linear = nn.Linear(self.config.hidden_size * 2, self.config.vocab_size)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        encoder_out, _ = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                   output_all_encoded_layers=False)
        output = self.transformer(encoder_out)
        gru_output, _ = self.gru(output)
        output = self.linear(gru_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(output.view(-1, self.config.vocab_size), labels.view(-1))
            return loss

        return output


class NezhaForSequenceClassificationGRUToAttention(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(NezhaForSequenceClassificationGRUToAttention, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 3, num_labels)
        self.apply(self.init_bert_weights)
        self.transformer = Transformer(max_len=512, hidden_size=config.hidden_size * 2)
        self.softmax = nn.Softmax(dim=-1)
        self.gru = nn.GRU(config.hidden_size, config.hidden_size, num_layers=2, bidirectional=True)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        encoder_out, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                               output_all_encoded_layers=False)
        gru_out, _ = self.gru(encoder_out)
        output = self.transformer(gru_out)
        resnet = torch.cat((pooled_output, output), axis=1)
        logits = self.classifier(resnet)
        logits = self.dropout(logits)
        prob = self.softmax(logits)
        if labels is not None:
            loss_fct = FocalLoss(class_num=self.num_labels)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits, prob


class NezhaForSequenceClassificationAttentionToGRU(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(NezhaForSequenceClassificationAttentionToGRU, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 3, num_labels)
        self.apply(self.init_bert_weights)
        self.transformer = Transformer(max_len=512, hidden_size=config.hidden_size)
        self.softmax = nn.Softmax(dim=-1)
        self.gru = nn.GRU(config.hidden_size, config.hidden_size, num_layers=2, bidirectional=True)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        encoder_out, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                               output_all_encoded_layers=False)
        output = self.transformer(encoder_out)
        gru_out, _ = self.gru(output)
        resnet = torch.cat((pooled_output, gru_out), axis=1)
        logits = self.classifier(resnet)
        logits = self.dropout(logits)
        prob = self.softmax(logits)
        if labels is not None:
            loss_fct = FocalLoss(class_num=self.num_labels)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits, prob
