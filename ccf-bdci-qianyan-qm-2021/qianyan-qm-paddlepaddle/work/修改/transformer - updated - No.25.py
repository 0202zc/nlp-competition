import paddle
import paddle.nn as nn
import math

class Transformer(nn.Layer):
    def __init__(self, max_len=512, hidden_size=768):
        super(Transformer, self).__init__()
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.head_nums = 8
        self.fc_1 = nn.Linear(768, hidden_size)
        self.fc_2 = nn.Linear(hidden_size, hidden_size * 4)
        self.fc_3 = nn.Linear(hidden_size * 4, hidden_size)
        self.fc_4 = nn.Linear(hidden_size, 2)
        self.sig = nn.Sigmoid()
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(-1)
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_size)
    def forward(self,x):
        x = self.fc_1(x)
        Q = paddle.transpose(self.q(x), perm=[0, 2 ,1])
        K = paddle.transpose(self.k(x), perm=[0, 2 ,1])
        V = paddle.transpose(self.v(x), perm=[0, 2 ,1])
        attention = paddle.transpose(paddle.matmul(self.softmax(paddle.matmul(Q, paddle.transpose(K, perm=[0, 2 ,1])) / math.sqrt(self.hidden_size)), V), perm=[0, 2 ,1])
        out = self.fc_2(attention)
        activate_out = self.gelu(out)
        out = self.fc_3(activate_out)
        out = self.layer_norm(out)
        out = self.fc_4(out)
        out = self.dropout(out)
        return paddle.mean(out, 1)
