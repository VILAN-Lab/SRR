import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import math
import copy
from torch import cosine_similarity as cs

USE_CUDA = torch.cuda.is_available()
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = 'cuda' if USE_CUDA else 'cpu'
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


def cross_attention(query, context, raw_feature_norm, smooth=9, eps=1e-8, weight=None):
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
    return attn_out, attnT


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
        pos_enc = get_sinusoid_encoding_table(seq_len, self.d_model).unsqueeze(0).to(device)
        pos_enc = pos_enc.repeat(batch_size, 1, 1)
        x = x + pos_enc
        # out = self.model(x)
        # for i in range(batch_size):
        #     out[i] = out[i][loc[i][0]] + out[i][loc[i][1]]
        # out = out.narrow(1, 1, 1)
        x = self.model(x)
        out = torch.zeros([batch_size, 1, 600], requires_grad=True).to(device)
        for i in range(batch_size):
            out[i] = torch.cat((x[i][loc[i][0]], x[i][loc[i][1]]))
        out = self.liner(out)
        return out


class AllModel(nn.Module):
    def __init__(self, words, weight):

        super(AllModel, self).__init__()
        self.SoftMax_2 = torch.nn.LogSoftmax(dim=2)
        # self.SoftMax_1 = torch.nn.Softmax(dim=1)
        self.De = OurModel().to(device)
        self.En = OurSREModel().to(device)
        with torch.no_grad():
            self.emb = torch.nn.Embedding(words, 300).to(device)
            self.emb.from_pretrained(weight, freeze=True)

    def forward(self, x, loc, ie):
        with torch.no_grad():
            x = self.emb(x)
        r = self.En(x, loc)
        out = self.De(r, ie)
        # out = self.SoftMax_1(out)
        out = self.SoftMax_2(out)
        return out
# a = AllModel()
# b = torch.Tensor(16,30,300)
# c = [[0,1]] * 16
# d = torch.Tensor(16, 5, 512)
# t = a(b,c,d)
# print(t.size())
# e = [[0,1,2,2,2]] *16
# e = torch.LongTensor(e)


def sre_loss_fn(train_out, target, threshold):
    batch_size = len(train_out)
    loss = torch.tensor([0]).to(device)
    for i in range(batch_size):
        loss = loss + 1 - cs(train_out[i], target[i])
    return loss


def loss_fn(out, target):
    # ob_num = 0.
    # irr_num = 0.
    # sub_num = 0.
    # batch_size = len(out)
    # loss = torch.zeros(1, requires_grad=True).to(device)
    # for i in range(batch_size):
    #     out[i], target[i]
    #     loss = loss + c(out[i], target[i])
    ob_loss = torch.zeros(1, requires_grad=True).to(device)
    sub_loss = torch.zeros(1, requires_grad=True).to(device)
    irr_loss = torch.zeros(1, requires_grad=True).to(device)

    for i in range(out.size(0)):
        for j in range(out.size(1)):
            if target[i, j] == 0:
                # ob_num += 1
                ob_loss += F.cross_entropy(out[i:i+1, j, :], target[i:i+1, j])
            elif target[i, j] == 1:
                sub_loss += F.cross_entropy(out[i:i+1, j, :], target[i:i+1, j])
                # sub_num += 1
            elif target[i, j] == 2:
                # irr_num += 1
                irr_loss += F.cross_entropy(out[i:i+1, j, :], target[i:i+1, j])
    # loss = F.cross_entropy(out, target)
    # loss = F.cross_entropy(out, target, reduction='sum')
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
