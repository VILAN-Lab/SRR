import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
import math
import copy
from torch import cosine_similarity as cs
from Models.GraphConvolution import GraphConvolution
from Models.graphSAGEmodels import *
from Models.utils import *
import time
USE_CUDA = torch.cuda.is_available()
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = 'cuda' if USE_CUDA else 'cpu'
cuda_0 = torch.device('cuda:0')
cuda_1 = torch.device('cuda:1')
# device = 'cpu'


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


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

    def forward(self, x, mask=None):
        x = self.sublayer[0](x, lambda x: self.attn(x, x, x, mask))
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

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q, k, v = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                   for l, x in zip(self.linears, (q, k, v))]
        x, self.attn = attention(q, k, v, mask, dropout=self.dp)
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


class OurModel(nn.Module):
    def __init__(self, n=3, d_model=300, d_ff=1200, h=6, dropout=0.3):
        super(OurModel, self).__init__()
        self.c = copy.deepcopy
        self.attn = MultiHeadAttention(h, d_model)
        self.ff = PosWiseFeedForward(d_model, d_ff, dropout)
        self.model = Encoder(EncoderLayer(d_model, self.c(self.attn), self.c(self.ff), dropout), n)
        self.ex = nn.Parameter(torch.Tensor(512, 300))
        torch.nn.init.uniform_(self.ex, -0.1, 0.1)
        self.cla = nn.Parameter(torch.Tensor(300, 3))
        torch.nn.init.uniform_(self.cla, -0.1, 0.1)

    def forward(self, r, ie):
        nums = len(ie[0])
        r = r.repeat(1, nums, 1)                             # batch, nums, temb
        ie = torch.matmul(ie, self.ex)
        r = r.permute(0, 2, 1)
        m = torch.matmul(ie, r)
        r = torch.matmul(r, m)
        r = r.permute(0, 2, 1)
        fusion = r + ie
        return torch.matmul(self.model(fusion), self.cla)


class LayerNorm1(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm1, self).__init__()
        self.a = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm1(layer.size)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class SubLayer1Connection(nn.Module):
    def __init__(self, size, dropout):
        super(SubLayer1Connection, self).__init__()
        self.norm = LayerNorm1(size)
        self.dp = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dp(sublayer(self.norm(x)))


class DecoderLayer(nn.Module):
    def __init__(self, size, attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.attn = attn
        self.ff = feed_forward
        self.size = size
        self.sublayer = clones(SubLayer1Connection(size, dropout), 2)

    def forward(self, x, mask=None):
        x = self.sublayer[0](x, lambda x: self.attn(x, x, x, mask))
        return self.sublayer[1](x, self.ff)


class DeMultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.3):
        super(DeMultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dp = nn.Dropout(dropout)
        self.attn = None

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q, k, v = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                   for l, x in zip(self.linears, (q, k, v))]
        x, self.attn = attention(q, k, v, mask, dropout=self.dp)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.linears[-1](x)


class DePosWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(DePosWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dp = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dp(F.relu(self.w_1(x))))


def get_sinusoid_encoding_table(n_position, d_hid):

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table)


class OurSREModel(nn.Module):
    def __init__(self, n=3, d_model=300, d_ff=1200, h=6, dropout=0.3):
        super(OurSREModel, self).__init__()
        self.d_model = d_model
        self.c = copy.deepcopy
        self.attn = DeMultiHeadAttention(h, d_model)
        self.ff = DePosWiseFeedForward(d_model, d_ff, dropout)
        self.model = Decoder(DecoderLayer(d_model, self.c(self.attn), self.c(self.ff), dropout), n)
        self.liner = nn.Linear(600, 300)

    def forward(self, x, loc):
        seq_len = len(x[0])
        batch_size = len(x)
        pos_enc = get_sinusoid_encoding_table(seq_len, self.d_model).unsqueeze(0).to(cuda_1)
        pos_enc = pos_enc.repeat(batch_size, 1, 1)
        x = x + pos_enc
        x = self.model(x)
        out = torch.zeros([batch_size, 1, 600], requires_grad=True).to(cuda_1)
        for i in range(batch_size):
            out[i] = torch.cat((x[i][loc[i][0]], x[i][loc[i][1]]))
            # out[i] = out[i][0]
        out = self.liner(out)
        return out


class GCNLayer(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCNLayer, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=True)
        x = self.gc2(x, adj)
        return x


def generate_adj(r):
    adj = euclidean_dist(r, r)
    zero = torch.zeros([adj.size(0), adj.size(1)], device=cuda_0)
    one = torch.ones([adj.size(0), adj.size(1)], device=cuda_0)
    adj = torch.where(adj > 3e-6, zero, one)
    adj = normalize(adj)
    return adj


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    x = x.double()
    y = y.double()
    # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    # yy会在最后进行转置的操作
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    # xy = x.mm(y.t())
    dist = xx + yy
    # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


class RelationGCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout):
        super(RelationGCN, self).__init__()
        self.GCNlayer_1 = GCNLayer(nfeat, nhid, nclass, dropout)
        self.GCNlayer_2 = GCNLayer(nfeat, nhid, nclass, dropout)

    def forward(self, r,):
        adj = generate_adj(r)
        r = self.GCNlayer_1(r, adj)
        adj = generate_adj(r)
        r = self.GCNlayer_1(r, adj)
        return r


class AllModel(nn.Module):
    def __init__(self, words, weight):

        super(AllModel, self).__init__()
        self.SoftMax_2 = torch.nn.LogSoftmax(dim=2).to(cuda_0)
        self.De = OurModel().to(cuda_0)
        self.Relationgcn = RelationGCN(300, 300, 300, 0.3).to(cuda_0)
        self.En = OurSREModel().to(cuda_1)
        self.emb = torch.nn.Embedding(words, 300, device=cuda_0)
        self.emb.from_pretrained(weight, freeze=self.training)

    def forward(self, x, loc, ie):
        x = self.emb(x)
        r = self.En(x.to(cuda_1), loc)
        r = r.squeeze(1)
        r = self.Relationgcn(r.to(cuda_0))
        r = r.unsqueeze(1)
        out = self.De(r, ie)
        out = self.SoftMax_2(out)
        return out


# def sre_loss_fn(train_out, target, threshold):
#     batch_size = len(train_out)
#     loss = torch.tensor([0], device=device)
#     for i in range(batch_size):
#         loss = loss + 1 - cs(train_out[i], target[i])
#     return loss


def loss_fn(out, target):
    ob_loss = torch.zeros(1, requires_grad=True, device=cuda_0)
    sub_loss = torch.zeros(1, requires_grad=True, device=cuda_0)
    irr_loss = torch.zeros(1, requires_grad=True, device=cuda_0)
    for i in range(out.size(0)):
        for j in range(out.size(1)):
            if target[i, j] == 0:
                ob_loss = ob_loss + F.cross_entropy(out[i:i+1, j, :], target[i:i+1, j])
            elif target[i, j] == 1:
                sub_loss = sub_loss + F.cross_entropy(out[i:i+1, j, :], target[i:i+1, j])
            elif target[i, j] == 2:
                irr_loss = irr_loss + F.cross_entropy(out[i:i+1, j, :], target[i:i+1, j])
    loss = (ob_loss + 40*sub_loss + 0.2 * irr_loss)/out.size(0)
    return loss
#
#
# q = loss_fn(t, e)
# print(q.item())

def test(out, target, length):
    m = []
    out = out.permute(0, 2, 1)
    out = out.detach().cpu().tolist()
    target = target.tolist()
    for i in range(len(out)):
        if length[i] == 1:
            if out[i][0].index(max(out[i][0])) == target[i].index(0):
                m.append(1)
            else:
                m.append(0)
        elif length[i] == 2:
            if out[i][0].index(max(out[i][0])) == target[i].index(0):
                m.append(1)
            else:
                m.append(0)
            if out[i][1].index(max(out[i][1])) == target[i].index(1):
                m.append(1)
            else:
                m.append(0)
    return sum(m) / len(m)
