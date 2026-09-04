import math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, head_size, num_heads, out_size, query_input_size=None):
        super(MultiHeadAttention, self).__init__()
        self.input_size = input_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.out_size = out_size
        self.query_input_size = self.input_size if query_input_size is None else query_input_size

        self.W_Q = nn.Linear(self.query_input_size, self.num_heads * self.head_size, bias=False)
        self.W_K = nn.Linear(self.input_size, self.num_heads * self.head_size, bias=False)
        self.W_V = nn.Linear(self.input_size, self.num_heads * self.head_size, bias=False)

        self.out = nn.Linear(self.head_size * self.num_heads, self.out_size)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, query, key, value, padding_mask=None):
        batch_size = key.size(0)
        emb_len = key.size(1)
        query_emb_len = query.size(1)

        q = self.W_Q(query)
        k = self.W_K(key)
        v = self.W_V(value)

        q = q.view(batch_size, query_emb_len, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(batch_size, emb_len, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, emb_len, self.num_heads, self.head_size).transpose(1, 2)

        k_T = k.transpose(2, 3)
        relevance = (q @ k_T) / math.sqrt(self.head_size)

        if padding_mask is not None:
            extended_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            relevance = relevance.masked_fill(extended_mask == 0, float('-inf'))

        relevance = F.softmax(relevance, dim=-1)
        relevance = self.dropout(relevance)

        heads = relevance @ v
        heads = heads.transpose(1, 2)
        concat = heads.reshape(batch_size, query_emb_len, self.head_size * self.num_heads)
        return self.out(concat)


class PositionalEncoding(nn.Module):
    def __init__(self, max_emb_len, d_model):
        super(PositionalEncoding, self).__init__()
        self.max_emb_len = max_emb_len
        self.d_model = d_model

        pos = torch.arange(max_emb_len)[:, None]
        i = torch.arange(d_model)[None, :]

        pe = torch.zeros(self.max_emb_len, self.d_model)
        sin = torch.sin(pos / (10000 ** (i[:, ::2] / self.d_model)))
        cos = torch.cos(pos / (10000 ** (i[:, 1::2] / self.d_model)))

        pe[:, ::2] = sin
        pe[:, 1::2] = cos

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, emb):
        emb_len = emb.size(1)
        emb = emb + self.pe[:, :emb_len]
        return self.dropout(emb)


class EncoderBlock(nn.Module):
    def __init__(self, input_size, head_size, num_heads, out_size, ff_hidden_size, query_input_size=None):
        super(EncoderBlock, self).__init__()
        self.input_size = input_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.out_size = out_size
        self.query_input_size = input_size if query_input_size is None else query_input_size
        self.ff_hidden_size = ff_hidden_size

        self.attention = MultiHeadAttention(
            input_size=self.input_size,
            head_size=self.head_size,
            num_heads=self.num_heads,
            out_size=self.out_size,
            query_input_size=self.query_input_size
        )

        if self.query_input_size != self.out_size:
            self.adapt = nn.Linear(self.query_input_size, self.out_size)
        else:
            self.adapt = nn.Identity()

        self.norm_1 = nn.LayerNorm(self.out_size)
        self.feed_forward = nn.Sequential(OrderedDict([
            ("Linear_1", nn.Linear(self.out_size, self.ff_hidden_size)),
            ("Activation", nn.ReLU()),
            ("Dropout", nn.Dropout(p=0.1)),
            ("Linear_2", nn.Linear(self.ff_hidden_size, self.out_size)),
        ]))
        self.norm_2 = nn.LayerNorm(self.out_size)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, query, key, value, padding_mask=None):
        attention_out = self.attention(query, key, value, padding_mask=padding_mask)
        attention_out = self.dropout(attention_out)

        add_1_out = attention_out + self.adapt(query)
        norm_1_out = self.norm_1(add_1_out)

        feed_forward_out = self.feed_forward(norm_1_out)
        feed_forward_out = self.dropout(feed_forward_out)

        add_out = feed_forward_out + norm_1_out
        return self.norm_2(add_out)


class TransformerEncoder(nn.Module):
    def __init__(self, N, max_seq_len, num_embeddings, emb_size, att_out_size, att_head_size, num_heads, ff_hidden_size):
        super(TransformerEncoder, self).__init__()
        self.N = N
        self.max_seq_len = max_seq_len
        self.num_embeddings = num_embeddings
        self.emb_size = emb_size
        self.att_out_size = att_out_size
        self.att_head_size = att_head_size
        self.num_heads = num_heads
        self.ff_hidden_size = ff_hidden_size

        self.embedding_layer = nn.Embedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.emb_size
        )
        self.positional_encoder = PositionalEncoding(
            max_emb_len=self.max_seq_len,
            d_model=self.emb_size
        )

        self.encoder_blocks = nn.ModuleDict({
            f"encoder_block_{i}": EncoderBlock(
                input_size=self.emb_size if i == 0 else self.att_out_size,
                head_size=self.att_head_size,
                num_heads=self.num_heads,
                out_size=self.att_out_size,
                ff_hidden_size=self.ff_hidden_size,
            ) for i in range(self.N)
        })

    def forward(self, encoder_input, padding_mask=None):
        encoder_emb = self.embedding_layer(encoder_input)
        out = self.positional_encoder(encoder_emb)

        for block in self.encoder_blocks.values():
            out = block(out, out, out, padding_mask=padding_mask)
        return out


class SentimentClassifier(nn.Module):
    def __init__(self, encoder, hidden_size, num_classes=2):
        super(SentimentClassifier, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, padding_mask):
        encoder_out = self.encoder(input_ids, padding_mask=padding_mask)
        mask_expanded = padding_mask.unsqueeze(-1).float()
        masked_embeddings = encoder_out * mask_expanded

        summed = torch.sum(masked_embeddings, dim=1)
        counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled_out = summed / counts

        return self.classifier(pooled_out)