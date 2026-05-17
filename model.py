import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, mask=None):

    d_k = Q.size(-1)

    scores = torch.matmul(Q, K.transpose(-2, -1))

    scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    attention_weights = F.softmax(scores, dim=-1)

    output = torch.matmul(attention_weights, V)

    return output, attention_weights


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model=128, num_heads=4):

        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, x):

        batch_size = x.size(0)

        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)

        Q = Q.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)

        attention_output, attention_weights = scaled_dot_product_attention(Q, K, V)

        attention_output = attention_output.transpose(1, 2).contiguous()

        attention_output = attention_output.view(
            batch_size,
            -1,
            self.d_model
        )

        output = self.out_linear(attention_output)

        return output, attention_weights


class SinusoidalPositionalEncoding(nn.Module):

    def __init__(self, d_model=128, max_len=5000):

        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):

        seq_len = x.size(1)

        return x + self.pe[:, :seq_len]


class LearnedPositionalEncoding(nn.Module):

    def __init__(self, d_model=128, max_len=5000):

        super().__init__()

        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):

        seq_len = x.size(1)

        positions = torch.arange(0, seq_len).to(x.device)

        positions = positions.unsqueeze(0)

        return x + self.embedding(positions)


class FeedForward(nn.Module):

    def __init__(self, d_model=128):

        super().__init__()

        self.linear1 = nn.Linear(d_model, d_model * 4)

        self.linear2 = nn.Linear(d_model * 4, d_model)

        self.relu = nn.ReLU()

    def forward(self, x):

        return self.linear2(self.relu(self.linear1(x)))


class EncoderLayer(nn.Module):

    def __init__(self, d_model=128, num_heads=4, dropout=0.1):

        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads)

        self.ffn = FeedForward(d_model)

        self.norm1 = nn.LayerNorm(d_model)

        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        attention_output, attention_weights = self.attention(x)

        x = self.norm1(x + self.dropout(attention_output))

        ffn_output = self.ffn(x)

        x = self.norm2(x + self.dropout(ffn_output))

        return x, attention_weights


class TransformerEncoder(nn.Module):

    def __init__(
        self,
        vocab_size=1000,
        d_model=128,
        num_heads=4,
        num_layers=2,
        max_len=100,
        num_classes=2
    ):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.position = SinusoidalPositionalEncoding(
            d_model,
            max_len
        )

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):

        attention_list = []

        x = self.embedding(x)

        x = self.position(x)

        for layer in self.layers:

            x, attention_weights = layer(x)

            attention_list.append(attention_weights)

        pooled = x.mean(dim=1)

        output = self.classifier(pooled)

        return output, attention_list