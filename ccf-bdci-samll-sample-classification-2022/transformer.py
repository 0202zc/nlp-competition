import torch
from torch import nn
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
