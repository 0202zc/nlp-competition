import paddle
import paddle.nn as nn
import math

class Transformer(nn.Layer):
    def __init__(self, max_len=512, hidden_size=768):
        super(Transformer, self).__init__()
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.head_nums = 8
        self.fc_1 = nn.Linear(hidden_size, hidden_size)
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
        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_size)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=4 * hidden_size)
        self.Tanh = nn.Tanh()

    def forward(self,x):
        x = self.fc_1(x)
        Q = paddle.transpose(self.q(x), perm=[0, 2 ,1])
        K = paddle.transpose(self.k(x), perm=[0, 2 ,1])
        V = paddle.transpose(self.v(x), perm=[0, 2 ,1])
        attention = paddle.transpose(paddle.matmul(self.softmax(paddle.matmul(Q, paddle.transpose(K, perm=[0, 2 ,1])) / math.sqrt(self.hidden_size)), V), perm=[0, 2 ,1])
        attention_out = self.fc_5(attention)
        attention_out = self.dropout(attention_out)
        out = self.fc_2(attention)
        activate_out = self.gelu(out)
        dropout_out = self.dropout(activate_out)
        out = self.layer_norm2(dropout_out)
        out = self.fc_3(out)
        out = self.dropout(out)
        out = self.layer_norm(out + attention_out)
        out = paddle.mean(out,1)
        out = self.fc_6(out)
        out = self.Tanh(out)
        return out
