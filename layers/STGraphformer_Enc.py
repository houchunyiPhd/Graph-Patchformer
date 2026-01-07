import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class Encoder(nn.Module):
    def __init__(self, attn_layers, patch_GCN, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.patch_GCN = nn.ModuleList(patch_GCN)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        adj_list = []
        for attn_layer, patch_GCN in zip(self.attn_layers, self.patch_GCN):
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)
            x, adj = patch_GCN(x)
            adj_list.append(adj)
        if self.norm is not None:
            x = self.norm(x)
        return x, adj_list

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )

        out = out.view(B, L, -1)

        return out, attn

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)  # 生成注意力分数

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))  # 将注意力分数通过缩放因子进行放缩
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None

class PgcnLayer(nn.Module):
    def __init__(self, GCN, d_model, dropout, batch_size, c_out, gcn=True):
        super(PgcnLayer, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        # self.dropout = nn.Dropout(dropout)
        # self.activation = nn.GELU()
        self.d_model = d_model
        self.gcn = GCN if gcn == True else None
        # self.Hypergcn = Hypergcnlayer()
        self.batch_size = batch_size
        self.c_out = c_out

    def forward(self, x):
        BN, P, _ = x.shape
        x = x.reshape(self.batch_size, self.c_out, P, -1).transpose(1, 2) # [B P N d_model] B C d_domel
        if self.gcn is not None:
           out, adj = self.gcn(x)
        return self.norm(out.reshape(BN, P, self.d_model) + x.reshape(BN, P, self.d_model)), adj

class Gcnlayer(nn.Module):
    def __init__(self, FullGNN, d_model, c_out, node_dim):
        super(Gcnlayer, self).__init__()

        self.nodevec1 = nn.Parameter(torch.randn(12, c_out, node_dim), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(12, node_dim, c_out), requires_grad=True)
        self.gnn = FullGNN
    def forward(self, x):
        adj = F.softmax(F.relu(torch.einsum('pnd,pde->pne',
                                            (self.nodevec1, self.nodevec2)
                                            )
                               ),
                        dim=-1)
        adj = adj + torch.eye(adj.shape[1]).to(x.device)
        d = adj.sum(dim=2)
        a = adj / d.view(x.shape[1], -1, 1)
        out = self.gnn(x, a)
        return out, a

class FullGNN(nn.Module):
    def __init__(self, k_hop, d_ff):
        super(FullGNN, self).__init__()
        self.nconv = nconv()
        self.k_hop = k_hop
        self.alpha = 0.5
        self.conv1 = nn.Conv2d(3, 1, kernel_size=1)
        self.flatten = nn.Flatten(start_dim=-2)
    def forward(self, x, A):
        h = x # [B P N d_moedl]
        out = [h.unsqueeze(1)]

        for i in range(self.k_hop):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, A)
            out.append(h.unsqueeze(1))
        ho = torch.cat(out, dim=1).transpose(2, 3)
        ho = self.flatten(ho)
        ho = self.conv1(ho).squeeze()
        ho = ho.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        return ho

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('bpnd,pne->bpnd', (x, A))
        return x.contiguous()