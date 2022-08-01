# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import paddlenlp as ppnlp

import transformer

class Embedding(nn.Layer):
    def __init__(self,max_len = 256,hidden_size = 768):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.vocab_size = 30000
        self.token_embedding = nn.Embedding(self.vocab_size,hidden_size)
        self.position_embedding = nn.Embedding(max_len,hidden_size)
        self.segment_embedding = nn.Embedding(2,hidden_size)
    def forward(self, input_ids, position_ids, segment_ids):
        # 不是预训练没有mask。
        input_ids = paddle.to_tensor(input_ids)
        position_ids = paddle.to_tensor(position_ids)
        segment_ids = paddle.to_tensor(segment_ids)
        token_emb = self.token_embedding(input_ids)
        position_emb = self.position_embedding(position_ids)
        segment_emb = self.segment_embedding(segment_ids)
        return token_emb + position_emb + segment_emb


class QuestionMatching(nn.Layer):
    def __init__(self, pretrained_model, dropout=None, rdrop_coef=0.0):
        super().__init__()
        self.ptm = pretrained_model
        self.vocab_size = self.ptm.config["vocab_size"]
        self.hidden_size = self.ptm.config["hidden_size"]
        self.emb_size = self.ptm.config["emb_size"]
        # self.embedding = paddle.nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.emb_size)
        self.embedding = Embedding()
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)

        self.transformer = transformer.Transformer(max_len=256, hidden_size=self.ptm.config["hidden_size"])
        # self.lstm = nn.LSTM(self.ptm.config["hidden_size"], self.ptm.config["hidden_size"], num_layers=2, direction='bidirectional')
        # self.lstm = nn.LSTM(self.ptm.config["hidden_size"], self.ptm.config["hidden_size"], num_layers=1)
        # self.classifier = nn.Sequential(
        #     nn.Linear(self.ptm.config["hidden_size"] * 2, 1024),
        #     nn.Linear(1024, 512),
        #     nn.Linear(512, 2))
        self.classifier = nn.Linear(self.ptm.config["hidden_size"] * 2, 2)
        self.fcnn = nn.Linear(self.ptm.config["hidden_size"] * 2, self.ptm.config["hidden_size"])
        self.rdrop_coef = rdrop_coef
        self.rdrop_loss = ppnlp.losses.RDropLoss()

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                do_evaluate=False):
        
        # 此处的 Input_ids 由两条文本的 token ids 拼接而成
        # token_type_ids 表示两段文本的类型编码
        # 返回的 cls_embedding 就表示这两段文本经过模型的计算之后而得到的语义表示向量

        # x, cls_embedding1 = self.ptm(input_ids, token_type_ids, position_ids, attention_mask)
        
        x, emb = self.embedding(input_ids, position_ids, token_type_ids)
        output = self.transformer(emb)
        '''
        output, (hidden, _) = self.lstm(x)

        # 将终态两个方向最后一层的隐状态拼接在一起 (Bi-LSTM)
        hidden = paddle.concat((hidden[-2, :, :], hidden[-1, :, :]), axis=1)
        hidden = self.fcnn(hidden)
        # 加入残差 x + f(x)
        resnet = paddle.concat((cls_embedding1, hidden), axis=1)  # Bi-LSTM
        # resnet = paddle.concat((cls_embedding1, hidden.mean(0)), axis=1)  # LSTM
        '''
        resnet = paddle.concat((emb, output), axis=1)
        logits1 = self.classifier(resnet)
        logits1 = self.dropout(logits1)
        
        # For more information about R-drop please refer to this paper: https://arxiv.org/abs/2106.14448
        # Original implementation please refer to this code: https://github.com/dropreg/R-Drop
        if self.rdrop_coef > 0 and not do_evaluate:
            _, cls_embedding2 = self.ptm(input_ids, token_type_ids, position_ids, attention_mask)
            cls_embedding2 = self.dropout(cls_embedding2)
            logits2 = self.classifier(cls_embedding2)
            kl_loss = self.rdrop_loss(logits1, logits2)
        else:
            kl_loss = 0.0

        return logits1, kl_loss
