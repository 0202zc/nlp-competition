import paddle
import paddle.nn as nn
import math


class Transformer(nn.Layer):
    def __init__(self, max_len=512, hidden_size=768):
        super(Transformer, self).__init__()
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.head_nums = 8
        self.fc_1 = nn.Linear(hidden_size * self.head_nums, hidden_size)
        self.fc_2 = nn.Linear(hidden_size, hidden_size * 4)
        self.fc_3 = nn.Linear(hidden_size * 4, hidden_size)
        self.fc_4 = nn.Linear(hidden_size, 1)
        self.sig = nn.Sigmoid()
        # 一个q有768*768个参数（fcnn），8个q就是8*768*768个参数
        self.q = [nn.Linear(hidden_size, hidden_size) for i in range(self.head_nums)]
        self.k = [nn.Linear(hidden_size, hidden_size) for i in range(self.head_nums)]
        self.v = [nn.Linear(hidden_size, hidden_size) for i in range(self.head_nums)]
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(-1)
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_size)

    def forward(self, x):
        res = []
        for i in range(self.head_nums):
            Q = paddle.transpose(self.q[i](x), perm=[0, 2 ,1])
            K = paddle.transpose(self.k[i](x), perm=[0, 2 ,1])
            V = paddle.transpose(self.v[i](x), perm=[0, 2 ,1])
            attention = paddle.transpose(paddle.matmul(self.softmax(paddle.matmul(Q, paddle.transpose(K, perm=[0, 2 ,1])) / math.sqrt(self.hidden_size)), V), perm=[0, 2 ,1])

            # Q = self.q[i](x).transpose(-2, -1)
            # K = self.k[i](x).transpose(-2, -1)
            # V = self.v[i](x).transpose(-2, -1)
            # attention = paddle.matmul(self.softmax(paddle.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.hidden_size)), V).transpose(-2, -1)
            res.append(attention)
        out = paddle.concat(res, -1)
        out = self.fc_1(out)
        out = self.fc_2(out)
        activate_out = self.gelu(out)
        out = self.fc_3(activate_out)
        out = self.layer_norm(out)
        return paddle.mean(out, 1)
