import torch
import os

from torch import nn
from focal_loss import FocalLoss
from modeling_nezha_small_sample import BertLayerNorm
from transformer import Transformer, TransformerForPretraining
from transformers import AutoModel, AutoConfig, AutoModelForSequenceClassification
from transformers.modeling_utils import SequenceSummary
from transformers import BertModel, BertPreTrainedModel
from transformers import NezhaModel, NezhaPreTrainedModel
from transformers import RobertaModel, RobertaPreTrainedModel
from transformers import XLNetPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from prefix_encoder import PrefixEncoder
from mixup import SenMixUp, MixUp

from tools.utils import torch_show_all_params

CONFIG_NAME = 'config.json'
WEIGHTS_NAME = 'pytorch_model.bin'


def torch_init_model(model, init_checkpoint, prefix='bert', delete_module=False):
    state_dict = torch.load(init_checkpoint, map_location='cpu')
    state_dict_new = {}
    # delete module.
    if delete_module:
        for key in state_dict.keys():
            v = state_dict[key]
            state_dict_new[key.replace('module.', '')] = v
        state_dict = state_dict_new
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})

        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix='' if hasattr(model, prefix) else str(prefix + '.'))

    print("missing keys:{}".format(missing_keys))
    print('unexpected keys:{}'.format(unexpected_keys))
    print('error msgs:{}'.format(error_msgs))


class XLNetPrefixForSequenceClassification(XLNetPreTrainedModel):
    def __init__(self, config, num_labels):
        super(XLNetPrefixForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.config = config
        self.hidden_size = config.d_model
        self.pre_seq_len = 6
        self.post_seq_len = 0

        self.enable_mixup = False

        if self.enable_mixup:
            self.mixup = MixUp(method='encoder')

        self.transformer = XLNetModel(config)
        # self.sequence_summary = SequenceSummary(config)
        # self.logits_proj = nn.Linear(self.hidden_size, self.num_labels)
        self.softmax = nn.Softmax(dim=-1)
        self.transformer_encoder = Transformer(max_len=486 + self.pre_seq_len + self.post_seq_len,
                                               hidden_size=self.hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.Linear(self.hidden_size // 2, self.num_labels)
        )
        self.dropout = nn.Dropout(0.1)
        # self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        # self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        # self.linear2 = nn.Linear(self.hidden_size * 2, self.hidden_size)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mems=None,
            perm_mask=None,
            target_mapping=None,
            token_type_ids=None,
            input_mask=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            use_mems=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            rdrop=False,
            **kwargs,  # delete when `use_cache` is removed in XLNetModel
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.shape[0]
        # past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.transformer.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        # input_ids = torch.cat((torch.ones(batch_size, self.pre_seq_len, dtype=torch.long).to(self.transformer.device), input_ids), dim=1)
        assert 0 < self.pre_seq_len < 100
        input_ids = torch.cat((torch.arange(1, self.pre_seq_len + 1, dtype=torch.long).unsqueeze(0).repeat(batch_size,
                                                                                                           1).to(
            self.transformer.device), input_ids), dim=1)
        token_type_ids = torch.cat(
            (torch.ones(batch_size, self.pre_seq_len, dtype=torch.long).to(self.transformer.device), token_type_ids),
            dim=1)

        # suffix_attention_mask = torch.ones(batch_size, self.post_seq_len).to(self.transformer.device)
        # attention_mask = torch.cat((attention_mask, suffix_attention_mask), dim=1)
        # assert 0 < self.post_seq_len < 100
        # input_ids = torch.cat((input_ids, torch.arange(self.pre_seq_len, self.pre_seq_len + self.post_seq_len,
        #                                                dtype=torch.long).unsqueeze(0).repeat(batch_size, 1).to(
        #     self.transformer.device)), dim=1)
        # token_type_ids = torch.cat(
        #     (token_type_ids, torch.ones(batch_size, self.post_seq_len, dtype=torch.long).to(self.transformer.device)),
        #     dim=1)

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_mems=use_mems,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        output = transformer_outputs[0]

        if self.enable_mixup:
            output = self.mixup.encode(output, [input_ids])

        cls_embedding = output[:, self.pre_seq_len, :]
        first_token_hidden_states = output[:, 0, :]
        last_token_hidden_states = output[:, -1, :]

        # lstm_output, _ = self.lstm(output)
        # gru_output, _ = self.gru(self.linear2(lstm_output))
        # gru_output, _ = self.gru(output)

        output = self.transformer_encoder(output)

        # output = self.sequence_summary(output)
        # logits = self.logits_proj(output)
        resnet = torch.cat((first_token_hidden_states, output), dim=1)
        # resnet = torch.cat((first_token_hidden_states, last_token_hidden_states, output), dim=1)
        logits = self.classifier(resnet)
        logits = self.dropout(logits)
        prob = self.softmax(logits)

        if labels is not None:
            if rdrop:
                from rdrop_loss import RDropLoss

                loss_fct = RDropLoss(class_num=self.num_labels)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                loss_fct = FocalLoss(class_num=self.num_labels)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss

        return logits, prob


class NezhaPrefixForPretraining(NezhaPreTrainedModel):
    def __init__(self, config):
        super(NezhaPrefixForPretraining, self).__init__(config)
        self.config = config
        self.dropout = nn.Dropout(0.2)
        self.nezha = NezhaModel(self.config)
        # self.gru = nn.GRU(self.config.hidden_size, self.config.hidden_size, num_layers=1, bidirectional=True)
        self.transformer1 = TransformerForPretraining(max_len=512, hidden_size=self.config.hidden_size)
        self.linear = nn.Linear(self.config.hidden_size, self.config.vocab_size)
        self.post_init()

        self.pre_seq_len = 16

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):
        batch_size = input_ids.shape[0]
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.nezha.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        assert 0 < self.pre_seq_len < 100
        input_ids = torch.cat((torch.arange(1, self.pre_seq_len + 1, dtype=torch.long).unsqueeze(0).repeat(batch_size,
                                                                                                           1).to(
            self.nezha.device), input_ids), dim=1)
        token_type_ids = torch.cat(
            (torch.ones(batch_size, self.pre_seq_len, dtype=torch.long).to(self.nezha.device), token_type_ids), dim=1)

        labels = torch.cat((torch.arange(1, self.pre_seq_len + 1, dtype=torch.long).unsqueeze(0).repeat(batch_size,
                                                                                                        1).to(
            self.nezha.device), labels), dim=1)

        outputs = self.nezha(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
            # past_key_values=past_key_values
        )

        # pooled_output = outputs[1]
        #
        # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)

        encoder_out = outputs[0]
        output = self.transformer1(encoder_out)
        output = self.linear(output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(output.view(-1, self.config.vocab_size), labels.view(-1))
            return loss

        return output


class NezhaForPretraining(BertPreTrainedModel):
    def __init__(self, bert_config):
        super(NezhaForPretraining, self).__init__(bert_config)
        self.config = bert_config
        self.dropout = nn.Dropout(0.2)
        self.bert = BertModel(self.config)
        self.apply(self.init_bert_weights)
        self.gru = nn.GRU(self.config.hidden_size, self.config.hidden_size, num_layers=1, bidirectional=True)
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


class NezhaForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(NezhaForSequenceClassification, self).__init__(config)
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


# class XLNetForSequenceClassification(XLNetPreTrainedModel):
#     def __init__(self, config, num_labels):
#         super().__init__(config)
#         self.num_labels = num_labels
#         self.config = config
#         self.hidden_size = config.d_model
#
#         self.transformer = XLNetModel(config)
#         self.sequence_summary = SequenceSummary(config)
#         self.logits_proj = nn.Linear(config.d_model, self.num_labels)
#         self.softmax = nn.Softmax(dim=-1)
#
#         # Initialize weights and apply final processing
#         self.post_init()
#
#     def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
#         transformer_outputs = self.transformer(input_ids=input_ids, token_type_ids=token_type_ids,
#                                                attention_mask=attention_mask)
#         output = transformer_outputs[0]
#         output = self.sequence_summary(output)
#         logits = self.logits_proj(output)
#         prob = self.softmax(logits)
#
#         if labels is not None:
#             loss_fct = FocalLoss(class_num=self.num_labels)
#             loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             return loss
#         return logits, prob


def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False


class SmallSampleClassification(nn.Module):
    def __init__(self, pretrained_model_path, max_seq_len, num_labels, freeze_embedding=False):
        super(SmallSampleClassification, self).__init__()
        self.num_labels = num_labels
        self.nezha = AutoModel.from_pretrained(pretrained_model_path)
        # torch_init_model(self.bert, os.path.join(pretrained_model_path, 'pytorch_model.bin'))
        # torch_show_all_params(self.bert)
        self.config = AutoConfig.from_pretrained(pretrained_model_path)
        self.hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.attention = Transformer(max_len=max_seq_len, hidden_size=self.hidden_size)
        # self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=1, bidirectional=True)
        self.classifier = nn.Linear(self.hidden_size * 2, num_labels)
        self.softmax = nn.Softmax(-1)
        self.apply(self.init_bert_weights)
        if freeze_embedding:
            freeze(self.nezha.embeddings)

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        ptm = self.nezha(input_ids, token_type_ids, attention_mask)
        encoder_out, pooled_out = ptm['last_hidden_state'], ptm['pooler_output']
        # gru_out, h_n = self.gru(encoder_out)
        output = self.attention(encoder_out)
        resnet = torch.cat((pooled_out, output), axis=1)
        logits = self.classifier(resnet)
        logits = self.dropout(logits)
        prob = self.softmax(logits)

        if labels is not None:
            loss_fct = FocalLoss(class_num=self.num_labels)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        return logits, prob


class AlteredBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = AlteredBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.transformer = Transformer(max_len=512, hidden_size=config.hidden_size)

        self.init_weights()

    def forward(
            self,
            inputs=None,
            inputs2=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            trace_grad=False,
            mixup_lambda=0,
            mixup_layer=-1,
            mix_embedding=False,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            inputs=inputs,
            inputs2=inputs2,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            trace_grad=trace_grad,
            mixup_lambda=mixup_lambda,
            mixup_layer=mixup_layer,
            mix_embedding=mix_embedding,
        )

        encoder_out, pooled_output = outputs[0], outputs[1]
        output = self.transformer(encoder_out)
        pooled_output = self.dropout(pooled_output)
        resnet = torch.cat((pooled_output, output), dim=1)

        logits = self.classifier(resnet)
        logits = self.dropout(logits)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                # loss_fct = nn.CrossEntropyLoss()
                loss_fct = FocalLoss(class_num=self.num_labels)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


from transformers import *
from transformers.models.bert.modeling_bert import (BertEmbeddings,
                                                    BertLayer,
                                                    BertPooler,
                                                    BertConfig,
                                                    BertPreTrainedModel,
                                                    SequenceClassifierOutput)
from transformers.modeling_outputs import BaseModelOutputWithPooling, BaseModelOutput


class AlteredBertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = AlteredBertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
            self,
            inputs=None,
            inputs2=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            trace_grad=False,
            mixup_lambda=0,
            mixup_layer=-1,
            mix_embedding=False,
    ):
        # Add exception for RoBERTa
        if inputs.get('token_type_ids') is None:
            inputs['token_type_ids'] = None
        if inputs2 is not None and inputs2.get('token_type_ids') is None:
            inputs2['token_type_ids'] = None

        input_ids, attention_mask, token_type_ids = inputs['input_ids'], inputs['attention_mask'], inputs[
            'token_type_ids']
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs2 is not None:
            input_ids2, attention_mask2, token_type_ids2 = inputs2['input_ids'], inputs2['attention_mask'], inputs2[
                'token_type_ids']
            input_shape2 = input_ids2.size()
            if attention_mask2 is None:
                attention_mask2 = torch.ones(input_shape2, device=device)
            if token_type_ids2 is None:
                token_type_ids = torch.zeros(input_shape2, dtype=torch.long, device=device)
            embedding_output2 = self.embeddings(
                input_ids=input_ids2, position_ids=position_ids, token_type_ids=token_type_ids2,
                inputs_embeds=inputs_embeds
            )
            extended_attention_mask2: torch.Tensor = self.get_extended_attention_mask(attention_mask2, input_shape2,
                                                                                      device)
        else:
            embedding_output2, extended_attention_mask2 = None, None

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        if trace_grad:
            embedding_output = embedding_output.detach().requires_grad_(True)
        if mix_embedding:
            embedding_output = mixup_lambda * embedding_output + (1 - mixup_lambda) * embedding_output2
            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask | attention_mask2,
                                                                                     input_shape, device)
            assert (mixup_layer == -1)

        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            hidden_states2=embedding_output2,
            attention_mask=extended_attention_mask,
            attention_mask2=extended_attention_mask2,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            mixup_lambda=mixup_lambda,
            mixup_layer=mixup_layer
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:] + (embedding_output,)

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions, )


class AlteredBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
            self,
            hidden_states=None,
            hidden_states2=None,
            attention_mask=None,
            attention_mask2=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
            mixup_lambda=0,
            mixup_layer=-1
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            # General step
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

            # step according to hidden layer mixup
            if mixup_layer != -1:
                assert (hidden_states2 is not None)
                assert (attention_mask2 is not None)

                if i <= mixup_layer:
                    layer_outputs2 = layer_module(hidden_states2, attention_mask2, layer_head_mask,
                                                  encoder_hidden_states,
                                                  encoder_attention_mask, output_attentions)
                    hidden_states2 = layer_outputs2[0]
                if i == mixup_layer:
                    hidden_states = mixup_lambda * hidden_states + (1 - mixup_lambda) * hidden_states2

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:  # We usually fall into this category
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertPrefixForPatentsClassification(BertPreTrainedModel):
    def __init__(self, config, num_class):
        super(BertPrefixForPatentsClassification, self).__init__(config)
        self.pre_seq_len = 8
        self.num_class = num_class
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2, num_class)
        self.transformer = Transformer(max_len=358 - self.pre_seq_len, hidden_size=config.hidden_size)
        # self.gru = nn.GRU(config.hidden_size, config.hidden_size, num_layers=1, bidirectional=True)
        self.softmax = nn.Softmax(dim=-1)
        self.rdrop = True
        # self.apply(self.init_bert_weights)
        self.post_init()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):

        batch_size = input_ids.shape[0]
        # past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        # input_ids = torch.cat((torch.ones(batch_size, self.pre_seq_len, dtype=torch.long).to(self.transformer.device), input_ids), dim=1)
        assert 0 < self.pre_seq_len < 100
        input_ids = torch.cat((torch.arange(1, self.pre_seq_len + 1, dtype=torch.long).unsqueeze(0).repeat(batch_size,
                                                                                                           1).to(
            self.bert.device), input_ids), dim=1)
        token_type_ids = torch.cat(
            (torch.ones(batch_size, self.pre_seq_len, dtype=torch.long).to(self.bert.device), token_type_ids),
            dim=1)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)

        encoder_out, pooled_output = outputs[0], outputs[1]
        output = self.transformer(encoder_out)
        resnet = torch.cat((pooled_output, output), dim=1)
        logits = self.classifier(resnet)
        logits = self.dropout(logits)
        prob = self.softmax(logits)

        if labels is not None:
            if self.rdrop:
                from rdrop_loss import RDropLoss

                loss_fct = RDropLoss(class_num=self.num_class)
                loss = loss_fct(logits.view(-1, self.num_class), labels.view(-1))
            else:
                loss_fct = FocalLoss(class_num=self.num_class)
                loss = loss_fct(logits.view(-1, self.num_class), labels.view(-1))
            return loss
        return logits, prob


class BertPrefixForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_class):
        super(BertPrefixForSequenceClassification, self).__init__(config)
        self.pre_seq_len = 8
        self.num_class = num_class
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2, num_class)
        self.transformer = Transformer(max_len=358 - self.pre_seq_len, hidden_size=config.hidden_size)
        # self.gru = nn.GRU(config.hidden_size, config.hidden_size, num_layers=1, bidirectional=True)
        self.softmax = nn.Softmax(dim=-1)
        # self.apply(self.init_bert_weights)
        self.post_init()

        # for param in self.bert.parameters():
        #     param.requires_grad = False

        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embed = config.hidden_size // config.num_attention_heads
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config, pre_seq_len=self.pre_seq_len, prefix_hidden_size=config.hidden_size,
                                            prefix_projection=True)

        bert_param = 0
        for name, param in self.bert.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('total param is {}'.format(total_param))  # 9860105

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embed
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):

        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values)

        # pooled_output = outputs[1]
        #
        # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)

        encoder_out, pooled_output = outputs[0], outputs[1]
        output = self.transformer(encoder_out)
        resnet = torch.cat((pooled_output, output), dim=1)
        logits = self.classifier(resnet)
        logits = self.dropout(logits)
        prob = self.softmax(logits)

        if labels is not None:
            # loss_fct = CrossEntropyLoss()
            loss_fct = FocalLoss(class_num=self.num_class)
            loss = loss_fct(logits.view(-1, self.num_class), labels.view(-1))
            # return logits, loss
            return loss
        return logits, prob


class BertPromptForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_class):
        super(BertPromptForSequenceClassification, self).__init__(config)
        self.bert = BertModel(config)
        self.embeddings = self.bert.embeddings
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2, num_class)
        self.transformer = Transformer(max_len=512, hidden_size=config.hidden_size)
        # self.gru = nn.GRU(config.hidden_size, config.hidden_size, num_layers=1, bidirectional=True)
        self.softmax = nn.Softmax(dim=-1)
        # self.apply(self.init_bert_weights)

        for param in self.bert.parameters():
            param.requires_grad = False

        self.pre_seq_len = 4
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embed = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = torch.nn.Embedding(self.pre_seq_len, config.hidden_size)

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        prompts = self.prefix_encoder(prefix_tokens)
        return prompts

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):
        batch_size, seq_length = input_ids.shape[0], input_ids.shape[1]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        raw_embedding = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
        prompts = self.get_prompt(batch_size=batch_size)
        inputs_embeds = torch.cat((prompts, raw_embedding), dim=1)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.bert(
            # input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            # position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # past_key_values=past_key_values
        )

        # encoder_out, pooled_output = outputs
        # output = self.transformer(encoder_out)
        # resnet = torch.cat((pooled_output, output), axis=1)
        # logits = self.classifier(resnet)
        # logits = self.dropout(logits)
        # prob = self.softmax(logits)

        sequence_output = outputs[0]
        sequence_output = sequence_output[:, self.pre_seq_len:, :].contiguous()
        first_token_tensor = sequence_output[:, 0]
        pooled_output = self.bert.pooler.dense(first_token_tensor)
        pooled_output = self.bert.pooler.activation(pooled_output)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        prob = self.softmax(logits)

        if labels is not None:
            # loss_fct = CrossEntropyLoss()
            loss_fct = FocalLoss(class_num=self.num_class)
            loss = loss_fct(logits.view(-1, self.num_class), labels.view(-1))
            # return logits, loss
            return loss
        return logits, prob


class NezhaPrefixForSequenceClassification(NezhaPreTrainedModel):
    def __init__(self, config, num_class):
        super(NezhaPreTrainedModel, self).__init__(config)
        self.num_class = num_class
        self.nezha = NezhaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2, num_class)
        # self.transformer1 = TransformerForPretraining(max_len=512, hidden_size=config.hidden_size)
        # self.transformer2 = Transformer(max_len=1, hidden_size=config.hidden_size * 2)
        # self.gru = nn.GRU(config.hidden_size, config.hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.transformer = Transformer(max_len=512, hidden_size=config.hidden_size)
        self.hidden_size = config.hidden_size
        self.softmax = nn.Softmax(dim=-1)
        # self.mixup = SenMixUp()
        # self.tanh = nn.Tanh()
        self.init_weights()

        # for param in self.nezha.parameters():
        #     param.requires_grad = False

        self.pre_seq_len = 16
        # self.n_layer = config.num_hidden_layers
        # self.n_head = config.num_attention_heads
        # self.n_embed = config.hidden_size // config.num_attention_heads
        # self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        # self.prefix_encoder = PrefixEncoder(config, pre_seq_len=self.pre_seq_len)
        #
        # bert_param = 0
        # for name, param in self.nezha.named_parameters():
        #     bert_param += param.numel()
        # all_param = 0
        # for name, param in self.named_parameters():
        #     all_param += param.numel()
        # total_param = all_param - bert_param
        # print('total param is {}'.format(total_param))  # 9860105

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.nezha.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embed
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):
        batch_size = input_ids.shape[0]
        # past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.nezha.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        # input_ids = torch.cat((torch.ones(batch_size, self.pre_seq_len, dtype=torch.long).to(self.nezha.device), input_ids), dim=1)
        assert 0 < self.pre_seq_len < 100
        input_ids = torch.cat((torch.arange(1, self.pre_seq_len + 1, dtype=torch.long).unsqueeze(0).repeat(batch_size,
                                                                                                           1).to(
            self.nezha.device), input_ids), dim=1)
        token_type_ids = torch.cat(
            (torch.ones(batch_size, self.pre_seq_len, dtype=torch.long).to(self.nezha.device), token_type_ids), dim=1)

        outputs = self.nezha(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
            # past_key_values=past_key_values
        )

        # pooled_output = outputs[1]
        #
        # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)

        encoder_out, pooled_output = outputs[0], outputs[1]

        encoder_pooled_out = self.transformer(encoder_out)
        resnet = torch.cat((encoder_pooled_out, pooled_output), dim=1)
        logits = self.classifier(resnet)
        logits = self.dropout(logits)
        prob = self.softmax(logits)

        # encoder_out = self.transformer2(encoder_out)
        # mixup_encoder_output, mixup_pooled_output = self.mixup.encoder(encoder_out, pooled_output)
        # output = torch.mean(self.tanh(mixup_encoder_output), 1)
        # resnet = torch.cat((mixup_pooled_output, output), dim=1)
        # logits = self.classifier(resnet)
        # logits = self.dropout(logits)
        # prob = self.softmax(logits)

        # encoder_out = self.transformer1(encoder_out)
        # gru_output, hn = self.gru(encoder_out)
        # last_hidden_state = torch.cat((gru_output[:, -1:, :self.hidden_size / 2], gru_output[:, :1, :self.hidden_size / 2]), dim=-1)
        # output = last_hidden_state.squeeze(1)
        # hn = torch.cat(hn.split(1), dim=-1).squeeze(0)
        # output = self.transformer2(hn.unsqueeze(1))
        # resnet = torch.cat((pooled_output, output), dim=1)
        # logits = self.classifier(resnet)
        # logits = self.dropout(logits)
        # prob = self.softmax(logits)

        if labels is not None:
            # loss_fct = CrossEntropyLoss()
            loss_fct = FocalLoss(class_num=self.num_class)
            loss = loss_fct(logits.view(-1, self.num_class), labels.view(-1))
            # loss = self.mixup(loss_fct, logits.view(-1, self.num_class), labels.view(-1))
            # return logits, loss
            return loss

        return logits, prob


class NezhaPrefixForSequenceClassificationBak(NezhaPreTrainedModel):
    def __init__(self, config, num_class):
        super(NezhaPreTrainedModel, self).__init__(config)
        self.num_class = num_class
        self.nezha = NezhaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2, num_class)
        self.transformer = Transformer(max_len=512, hidden_size=config.hidden_size)
        self.hidden_size = config.hidden_size
        # self.gru = nn.GRU(config.hidden_size, config.hidden_size, num_layers=1, bidirectional=True)
        self.softmax = nn.Softmax(dim=-1)
        # self.apply(self.init_bert_weights)

        for param in self.nezha.parameters():
            param.requires_grad = False

        self.pre_seq_len = 4
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embed = config.hidden_size // config.num_attention_heads
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config, pre_seq_len=self.pre_seq_len)

        bert_param = 0
        for name, param in self.nezha.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('total param is {}'.format(total_param))  # 9860105

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.nezha.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embed
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):

        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.nezha.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.nezha(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values)

        encoder_out, pooled_output = outputs[0], outputs[1]
        output = self.transformer(encoder_out)
        resnet = torch.cat((pooled_output, output), axis=1)
        logits = self.classifier(resnet)
        logits = self.dropout(logits)
        prob = self.softmax(logits)

        if labels is not None:
            # loss_fct = CrossEntropyLoss()
            loss_fct = FocalLoss(class_num=self.num_class)
            loss = loss_fct(logits.view(-1, self.num_class), labels.view(-1))
            # return logits, loss
            return loss
        return logits, prob


class NezhaPromptForSequenceClassification(NezhaPreTrainedModel):
    def __init__(self, config, num_class):
        super(NezhaPromptForSequenceClassification, self).__init__(config)
        self.nezha = NezhaModel(config)
        self.embeddings = self.nezha.embeddings
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2, num_class)
        self.transformer = Transformer(max_len=512, hidden_size=config.hidden_size)
        # self.gru = nn.GRU(config.hidden_size, config.hidden_size, num_layers=1, bidirectional=True)
        self.softmax = nn.Softmax(dim=-1)
        # self.apply(self.init_bert_weights)

        for param in self.nezha.parameters():
            param.requires_grad = False

        self.pre_seq_len = 4
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embed = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = torch.nn.Embedding(self.pre_seq_len, config.hidden_size)

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.nezha.device)
        prompts = self.prefix_encoder(prefix_tokens)
        return prompts

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):
        batch_size, seq_length = input_ids.shape[0], input_ids.shape[1]
        raw_embedding = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )
        prompts = self.get_prompt(batch_size=batch_size)
        inputs_embeds = torch.cat((prompts, raw_embedding), dim=1)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.nezha.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.nezha(
            # input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # past_key_values=past_key_values
        )

        encoder_out, pooled_output = outputs
        output = self.transformer(encoder_out)
        resnet = torch.cat((pooled_output, output), dim=1)
        logits = self.classifier(resnet)
        logits = self.dropout(logits)
        prob = self.softmax(logits)

        if labels is not None:
            # loss_fct = CrossEntropyLoss()
            loss_fct = FocalLoss(class_num=self.num_class)
            loss = loss_fct(logits.view(-1, self.num_class), labels.view(-1))
            # return logits, loss
            return loss
        return logits, prob


class RobertaPrefixForSequenceClassification(RobertaPreTrainedModel):
    def __init__(self, config, num_class):
        super().__init__(config)
        self.num_labels = num_class
        self.config = config
        self.pre_seq_len = 16
        self.bert = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.transformer = Transformer(max_len=512, hidden_size=config.hidden_size)
        self.softmax = nn.Softmax(dim=-1)
        self.init_weights()

        # for param in self.bert.parameters():
        #     param.requires_grad = False
        #
        # self.n_layer = config.num_hidden_layers
        # self.n_head = config.num_attention_heads
        # self.n_embd = config.hidden_size // config.num_attention_heads
        #
        # self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        # self.prefix_encoder = PrefixEncoder(config)
        #
        # bert_param = 0
        # for name, param in self.bert.named_parameters():
        #     bert_param += param.numel()
        # all_param = 0
        # for name, param in self.named_parameters():
        #     all_param += param.numel()
        # total_param = all_param - bert_param
        # print('total param is {}'.format(total_param))  # 9860105

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        batch_size = input_ids.shape[0]
        # past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        input_ids = torch.cat(
            (torch.ones(batch_size, self.pre_seq_len, dtype=torch.long).to(self.bert.device), input_ids), dim=1)
        token_type_ids = torch.cat(
            (torch.ones(batch_size, self.pre_seq_len, dtype=torch.long).to(self.bert.device), token_type_ids), dim=1)
        position_ids = torch.cat((position_ids,
                                  torch.arange(512 - self.pre_seq_len, 512, dtype=torch.long).unsqueeze(0).repeat(
                                      batch_size, 1).to(
                                      self.bert.device)), dim=1)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
            # past_key_values=past_key_values,
        )

        # pooled_output = outputs[1]
        #
        # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)

        encoder_out, pooled_output = outputs[0], outputs[1]
        output = self.transformer(encoder_out)
        resnet = torch.cat((pooled_output, output), dim=1)
        logits = self.classifier(resnet)
        logits = self.dropout(logits)
        prob = self.softmax(logits)

        if labels is not None:
            # loss_fct = CrossEntropyLoss()
            loss_fct = FocalLoss(class_num=self.num_labels)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            # return logits, loss
            return loss
        return logits, prob


class RobertaPrefixForSequenceClassificationBak(RobertaPreTrainedModel):
    def __init__(self, config, num_class):
        super().__init__(config)
        self.num_labels = num_class
        self.config = config
        self.pre_seq_len = 4
        self.bert = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.transformer = Transformer(max_len=512 - self.pre_seq_len, hidden_size=config.hidden_size)
        self.softmax = nn.Softmax(dim=-1)
        self.init_weights()

        for param in self.bert.parameters():
            param.requires_grad = False

        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config)

        bert_param = 0
        for name, param in self.bert.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('total param is {}'.format(total_param))  # 9860105

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):

        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )

        # pooled_output = outputs[1]
        #
        # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)

        encoder_out, pooled_output = outputs[0], outputs[1]
        output = self.transformer(encoder_out)
        resnet = torch.cat((pooled_output, output), dim=1)
        logits = self.classifier(resnet)
        logits = self.dropout(logits)
        prob = self.softmax(logits)

        if labels is not None:
            # loss_fct = CrossEntropyLoss()
            loss_fct = FocalLoss(class_num=self.num_labels)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            # return logits, loss
            return loss
        return logits, prob


class RobertaPromptForSequenceClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.embeddings = self.roberta.embeddings
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

        for param in self.roberta.parameters():
            param.requires_grad = False

        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = torch.nn.Embedding(self.pre_seq_len, config.hidden_size)

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.roberta.device)
        prompts = self.prefix_encoder(prefix_tokens)
        return prompts

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.shape[0]
        raw_embedding = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
        prompts = self.get_prompt(batch_size=batch_size)
        inputs_embeds = torch.cat((prompts, raw_embedding), dim=1)
        # print(input_embeddings.shape)
        # exit()
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.roberta.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.roberta(
            # input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            # position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # past_key_values=past_key_values,
        )

        # pooled_output = outputs[1]
        sequence_output = outputs[0]
        sequence_output = sequence_output[:, self.pre_seq_len:, :].contiguous()
        first_token_tensor = sequence_output[:, 0]
        pooled_output = self.roberta.pooler.dense(first_token_tensor)
        pooled_output = self.roberta.pooler.activation(pooled_output)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


if __name__ == '__main__':
    model_path = 'pretrained_models/nezha-base-wwm'
    model = SmallSampleClassification(model_path, 512, 36, freeze_embedding=True)
    # ptm = AutoModel.from_pretrained('pretrained_models/nezha-base-wwm')
    # print(ptm.state_dict().keys())

    torch_init_model(model, os.path.join(model_path, 'pytorch_model.bin'), prefix='nezha')
    print(model.state_dict().keys())

    # torch_init_model(model, os.path.join('pretrained_models/nezha-base-wwm', 'pytorch_model.bin'), prefix='nezha')
    # torch_show_all_params(model)
    # print(list(model.named_parameters()))
    # print("********************************************")
    # print([(k, v) for k, v in list(model.named_parameters()) if not v.requires_grad])
    # print("********************************************")
    # print([p for p in list(model.parameters()) if p.requires_grad])
    # print(model.parameters())
