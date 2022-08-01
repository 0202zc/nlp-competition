import paddle
import paddle.nn as nn
import math

class Transformer(nn.Layer):
    def __init__(self, max_len=256, hidden_size=768):
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
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=hidden_size*4)
        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_size)
        self.Tanh = nn.Tanh()
    def forward(self, x):
        self.max_len = x.shape[1]
        Q = paddle.transpose(paddle.reshape(self.q(x), [-1, self.max_len, self.head_nums, self.hidden_size // self.head_nums]), perm=[0, 2, 3, 1])
        K = paddle.transpose(paddle.reshape(self.k(x), [-1, self.max_len, self.head_nums, self.hidden_size // self.head_nums]), perm=[0, 2, 3, 1])
        V = paddle.transpose(paddle.reshape(self.v(x), [-1, self.max_len, self.head_nums, self.hidden_size // self.head_nums]), perm=[0, 2, 3, 1])
        attention = paddle.transpose(paddle.matmul(self.softmax(paddle.matmul(Q, paddle.transpose(K, perm=[0, 1, 3, 2])) / math.sqrt(self.hidden_size // self.head_nums)), V), perm=[0, 1, 3, 2])
        attention = paddle.transpose(attention, perm=[0, 2, 1, 3])
        attention = paddle.reshape(attention, [-1, self.max_len, self.hidden_size])
        attention_out = self.fc_5(attention)
        attention_out = self.dropout(attention_out)
        out = self.fc_2(attention)
        activate_out = self.gelu(out)
        dropout_out = self.dropout(activate_out)
        out = self.layer_norm_2(dropout_out)
        out = self.fc_3(out)
        out = self.dropout(out)
        out = self.layer_norm(out + attention_out)
        out = paddle.mean(out, 1)
        out = self.fc_6(out)
        out = self.Tanh(out)
        #out = self.fc_4(out)
        #out = self.dropout(out)
        return out
