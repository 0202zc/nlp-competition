import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):
    """
    input : batch_size * seq_len * input_dim
    q : batch_size * input_dim * dim_k
    k : batch_size * input_dim * dim_k
    v : batch_size * input_dim * dim_v
    """

    def __init__(self, input_dim, dim_k, dim_v):
        super().__init__()
        self.dim_k = dim_k
        self.q = nn.Linear(input_dim, dim_k)
        self.k = nn.Linear(input_dim, dim_k)
        self.v = nn.Linear(input_dim, dim_v)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.q(x)  # Q: batch_size * seq_len * dim_k
        K = self.k(x)  # K: batch_size * seq_len * dim_k
        V = self.v(x)  # V: batch_size * seq_len * dim_v

        attention = torch.bmm(self.softmax(torch.bmm(Q, K.permute(0, 2, 1)) / math.sqrt(self.dim_k)), V)

        return attention


class MultiHeadSelfAttention(nn.Module):
    """
    input : batch_size * seq_len * input_dim
    q : batch_size * input_dim * dim_k
    k : batch_size * input_dim * dim_k
    v : batch_size * input_dim * dim_v
    """

    def __init__(self, input_dim, dim_k, dim_v, nums_head):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k % nums_head == 0
        assert dim_v % nums_head == 0
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.q = nn.Linear(input_dim, dim_k)
        self.k = nn.Linear(input_dim, dim_k)
        self.v = nn.Linear(input_dim, dim_v)

        self.nums_head = nums_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.q(x).view(-1, x.shape[1], self.nums_head, self.dim_k // self.nums_head).permute(0, 2, 3, 1)
        K = self.k(x).view(-1, x.shape[1], self.nums_head, self.dim_k // self.nums_head).permute(0, 2, 3, 1)
        V = self.v(x).view(-1, x.shape[1], self.nums_head, self.dim_v // self.nums_head).permute(0, 2, 3, 1)

        attention = torch.matmul(self.softmax(torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(self.dim_k)),
                                 V).transpose(-2, -1)  # [batch_size, n_head, seq_len, hidden_size // n_head]
        attention = attention.transpose(1, 2)  # [batch_size, seq_len, n_head, hidden_size // n_head]

        output = attention.reshape(-1, x.shape[1], x.shape[2])

        # attention = attention.permute(2, 0, 1, 3)
        # output = torch.cat([_ for _ in attention], dim=-1)

        return output


if __name__ == '__main__':
    torch.manual_seed(42)
    a = torch.randn((1, 3, 4))
    attention2 = SelfAttention(4, 4, 4)
    attention = MultiHeadSelfAttention(4, 4, 4, 2)
    out = attention(a)
    out2 = attention2(a)
    print(out.shape, out2.shape)
    print(out)
