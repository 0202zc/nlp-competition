import torch
from torch import nn
from torch.nn import functional as F
import math


class TransformerForPretraining(nn.Module):
    def __init__(self, max_len=512, hidden_size=768):
        super(TransformerForPretraining, self).__init__()
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.head_nums = 8
        self.fc_1 = nn.Linear(hidden_size, hidden_size)  # 将您上一层的输出转为本层的hidden_size，谢您嘞！
        self.fc_2 = nn.Linear(hidden_size, hidden_size * 4)
        self.fc_3 = nn.Linear(hidden_size * 4, hidden_size)
        self.fc_4 = nn.Linear(hidden_size, 2)
        self.fc_5 = nn.Linear(hidden_size, hidden_size)
        self.fc_6 = nn.Linear(hidden_size, hidden_size)
        self.sig = nn.Sigmoid()
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(-1)
        self.gelu = nn.GELU()
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=hidden_size * 4)
        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_size)
        self.Tanh = nn.Tanh()

    def forward(self, x):
        Q = self.q(x).reshape(-1, self.max_len, self.head_nums, self.hidden_size // self.head_nums).permute(0, 2, 3, 1)
        K = self.k(x).reshape(-1, self.max_len, self.head_nums, self.hidden_size // self.head_nums).permute(0, 2, 3, 1)
        V = self.v(x).reshape(-1, self.max_len, self.head_nums, self.hidden_size // self.head_nums).permute(0, 2, 3, 1)
        attention = torch.matmul(
            self.softmax(torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.hidden_size // self.head_nums)),
            V).transpose(-2, -1)
        attention = attention.transpose(1, 2)
        attention = attention.reshape(-1, self.max_len, self.hidden_size)
        attention_out = self.fc_5(attention)
        attention_out = self.dropout(attention_out)
        out = self.fc_2(attention)
        activate_out = self.gelu(out)
        dropout_out = self.dropout(activate_out)
        out = self.layer_norm_2(self.fc_2(x) + dropout_out)
        out = self.fc_3(out)
        out = self.dropout(out)
        out = self.layer_norm(out + attention_out)
        return out


class Transformer(nn.Module):
    def __init__(self, max_len=256, hidden_size=768):
        super(Transformer, self).__init__()
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.head_nums = 8
        self.fc_1 = nn.Linear(hidden_size, hidden_size)  # 将您上一层的输出转为本层的hidden_size，谢您嘞！
        self.fc_2 = nn.Linear(hidden_size, hidden_size * 4)
        self.fc_3 = nn.Linear(hidden_size * 4, hidden_size)
        self.fc_4 = nn.Linear(hidden_size, 2)
        self.fc_5 = nn.Linear(hidden_size, hidden_size)
        self.fc_6 = nn.Linear(hidden_size, hidden_size)
        self.sig = nn.Sigmoid()
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(-1)
        self.gelu = nn.GELU()
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=hidden_size * 4)
        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_size)
        self.Tanh = nn.Tanh()

    def forward(self, x):
        Q = self.q(x).reshape(-1, self.max_len, self.head_nums, self.hidden_size // self.head_nums).permute(0, 2, 3, 1)
        K = self.k(x).reshape(-1, self.max_len, self.head_nums, self.hidden_size // self.head_nums).permute(0, 2, 3, 1)
        V = self.v(x).reshape(-1, self.max_len, self.head_nums, self.hidden_size // self.head_nums).permute(0, 2, 3, 1)
        attention = torch.matmul(
            self.softmax(torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.hidden_size // self.head_nums)),
            V).transpose(-2, -1)
        attention = attention.transpose(1, 2)
        attention = attention.reshape(-1, self.max_len, self.hidden_size)
        attention_out = self.fc_5(attention)
        attention_out = self.dropout(attention_out)
        out = self.fc_2(attention)
        activate_out = self.gelu(out)
        dropout_out = self.dropout(activate_out)
        out = self.layer_norm_2(self.fc_2(x) + dropout_out)
        out = self.fc_3(out)
        out = self.dropout(out)
        out = self.layer_norm(out + attention_out)
        out = torch.mean(out, 1)
        out = self.fc_6(out)
        out = self.Tanh(out)
        # out = self.fc_4(out)
        # out = self.dropout(out)
        return out


# 测试
# x = torch.randn([2,256,768])
# net = Transformer()
# print(net(x))


class Attention(nn.Module):
    """注意力层。"""

    def __init__(self, hidden_size, **kwargs):
        self.hidden_size = hidden_size
        super().__init__(**kwargs)
        self.weight = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.hidden_size))
        self.query = nn.Linear(self.hidden_size, 1, bias=False)

    def forward(self, x, mask):
        '''x: [btz, max_segment, hdsz]
        mask: [btz, max_segment, 1]
        '''
        mask = mask.squeeze(2)  # [btz, max_segment]

        # linear
        key = self.weight(x) + self.bias  # [btz, max_segment, hdsz]

        # compute attention
        outputs = self.query(key).squeeze(2)  # [btz, max_segment]
        outputs -= 1e32 * (1 - mask)
        attn_scores = F.softmax(outputs, dim=-1)
        attn_scores = attn_scores * mask
        attn_scores = attn_scores.reshape(-1, 1, attn_scores.shape[-1])  # [btz, 1, max_segment]

        outputs = torch.matmul(attn_scores, key).squeeze(1)  # [btz, hdsz]
        return outputs


"""
# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, segment_vocab_size=0)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.attn = Attention(768)
        self.dense = nn.Linear(768, num_classes)

    def forward(self, token_ids):
        ''' token_ids: [btz, max_segment, max_len]
        '''
        input_mask = torch.any(token_ids, dim=-1, keepdim=True).long()  # [btz, max_segment, 1]
        token_ids = token_ids.reshape(-1, token_ids.shape[-1])  # [btz*max_segment, max_len]

        output = self.bert([token_ids])[:, 0]  # [btz*max_segment, hdsz]
        output = output.reshape((-1, max_segment, output.shape[-1]))  # [btz, max_segment, hdsz]
        output = output * input_mask
        output = self.dropout1(output)
        output = self.attn(output, input_mask)
        output = self.dropout2(output)
        output = self.dense(output)
        return output
"""
