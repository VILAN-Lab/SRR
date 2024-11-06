import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import random
import numpy as np


device = 'cuda' if torch.cuda.is_available() else "cpu"
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def cross_attention(query, context, raw_feature_norm="softmax", smooth=9, eps=1e-8, weight=None):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    weight: (n_context, sourceL, queryL)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)

    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)

    if raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size*sourceL, queryL)
        attn = F.softmax(attn, dim=1)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif raw_feature_norm == "l2norm":
        attn = nn.norm(attn, 2)
    elif raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = nn.norm(attn, 2)
    elif raw_feature_norm == "l1norm":
        attn = nn.norm(attn, 2)
    elif raw_feature_norm == "clipped_l1norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = nn.norm(attn, 2)
    elif raw_feature_norm == "clipped":
        attn = nn.LeakyReLU(0.1)(attn)
    elif raw_feature_norm == "no_norm":
        pass
    else:
        raise ValueError("unknown first norm type:", raw_feature_norm)

    if weight is not None and weight.size(1) == attn.size(1) and weight.size(2) == attn.size(2):
      attn = attn*(1-weight)

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    cosines_attn = attn.max(1)[0]
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size*queryL, sourceL)

    attn = F.softmax(attn*smooth, dim=1)
    attn_out = attn.clone()
    attn_out = attn_out.view(batch_size, sourceL, queryL)   # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    return torch.matmul(attn_out, query).sum(1).unsqueeze(1), torch.transpose(attnT, 1, 2)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class SubLayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SubLayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dp = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dp(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.attn = attn
        self.ff = feed_forward
        self.size = size
        self.sublayer = clones(SubLayerConnection(size, dropout), 2)

    def forward(self, x,i, mask=None):
        x = self.sublayer[0](x, lambda x: self.attn(x, x, x, i, mask))
        return self.sublayer[1](x, self.ff)


def attention(q, k, v, mask=None, dropout=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, v), p_attn



class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.3):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dp = nn.Dropout(dropout)
        self.attn = None

    def forward(self, q, k, v, i, mask=None):
        batch_size = q.size(0)
        q, k, v = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                   for l, x in zip(self.linears, (q, k, v))]
        x, self.attn = cross_attention(i, v)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PosWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(PosWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dp = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dp(F.relu(self.w_1(x))))


class Trans(nn.Module):
    def __init__(self, n=3, d_model=300, d_ff=1200, h=6, dropout=0.3):
        super(Trans, self).__init__()
        self.c = copy.deepcopy
        self.attn = MultiHeadAttention(h, d_model)
        self.ff = PosWiseFeedForward(d_model, d_ff, dropout)
        self.model = Encoder(EncoderLayer(d_model, self.c(self.attn), self.c(self.ff), dropout), n)

    def forward(self, x, i):
        return self.model(x, i)

"""
text-attention map
"""


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.Tensor(in_features, out_features))
        self.a = nn.Parameter(torch.Tensor(2 * out_features, 1))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        '''
        h: (N, in_features)
        adj: sparse matrix with shape (N, N)
        '''
        Wh = torch.mm(h, self.W)  # (N, out_features)
        Wh1 = torch.mm(Wh, self.a[:self.out_features, :])  # (N, 1)
        Wh2 = torch.mm(Wh, self.a[self.out_features:, :])  # (N, 1)
        e = self.leakyrelu(Wh1 + Wh2.T)  # (N, N)
        padding = (-2 ** 31) * torch.ones_like(e)  # (N, N)
        attention = torch.where(adj > 0, e, padding)  # (N, N)
        attention = F.softmax(attention, dim=1)  # (N, N)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, Wh)  # (N, out_features)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime





class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.MH = nn.ModuleList([
            GraphAttentionLayer(nfeat, nhid, dropout, alpha, concat=True)
            for _ in range(nheads)
        ])
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout, alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)  # (N, nfeat)
        x = torch.cat([head(x, adj) for head in self.MH], dim=1)  # (N, nheads*nhid)
        x = F.dropout(x, self.dropout, training=self.training)  # (N, nheads*nhid)
        x = F.elu(self.out_att(x, adj))
        return x

#
# a = GAT(500, 300, 300, 0.3, 0, 3)
# ins = torch.Tensor(50, 500)
# adj = torch.Tensor(50, 50)
# c = a(ins, adj)
# print(c.size())


class Gen(nn.Module):
    def __init__(self, vocab_size, embed_size, hiddens_size, num_layers):
        super(Gen, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hiddens_size, num_layers)
        self.linear = nn.Linear(hiddens_size, vocab_size)

    def forward(self, x, s):
        out, sta = self.rnn(x)
        out = out.sum(1)
        return self.linear(out), sta


# m = Gen(400000, 300, 300, 2)
# a = torch.randn(1, 15, 300)
# b,c = m(a)
# print(b.size())
# print(c.size())


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(1024,300)
        self.trans = Trans()
        self.gat = GAT(1024, 1024, 300, 0.3, 0, 3)
        self.gen = Gen(400000, 300, 300, 2)

    def forward(self, t, i, adj):
        ti = self.linear(i)
        img_fea = self.gat(i, adj)
        t_fea = self.trans(t, ti)
        img_fea = (img_fea.sum(0))/len(img_fea)
        img_fea = img_fea.unsqueeze(0)
        fusion_fea = img_fea + t_fea
        # print(fusion_fea.size(1))
        s_0 = torch.zeros(2, fusion_fea.size(1), 300).to(device)
        outputs_save = torch.zeros(1, 20, 400000).to(device)
        out, s = self.gen(fusion_fea, s_0)
        outputs_save[:, 0, :] = out
        # print(outputs_save)
        for j in range(1, 20):
            out, s = self.gen(fusion_fea, s)
            outputs_save[:, j, :] = out
        return outputs_save

#
# m = Model()
# a = torch.randn(1, 15, 300)
# b = torch.randn(64, 1024)
# c = torch.randn(64, 64)
# print(m(a, b, c).size())
