import torch
import torch.nn as nn


class DataEmbedding_inverted_patching(nn.Module):
    def __init__(self, c_in, seq_len, d_model, patch_len, stride, padding, dropout):
        super(DataEmbedding_inverted_patching, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        # node embedding
        self.node_emb = nn.Parameter(torch.empty(c_in, seq_len))
        nn.init.xavier_uniform_(self.node_emb)

        # hour embedding
        self.hour_embed = nn.Parameter(torch.empty(24, seq_len))  # [hour d_model]
        nn.init.xavier_uniform_(self.hour_embed)

        # day embedding
        self.day_embed = nn.Parameter(torch.empty(7, seq_len))  # [day d_model]
        nn.init.xavier_uniform_(self.day_embed)

        # high-dimension mapping
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark):
        x = x.transpose(1, 2)  # [B N T]
        B, N, T = x.shape

        node_emb = self.node_emb.unsqueeze(0).expand(B, -1, -1)  # [B N d_model]
        if x_mark.shape[-1] >= 2:
            hour_embed = self.hour_embed[(x_mark[:, :, 0].unsqueeze(-1).expand(-1, -1, N)[:, -1, :]).type(torch.LongTensor)]
            day_embed = self.day_embed[(x_mark[:, :, 1].unsqueeze(-1).expand(-1, -1, N)[:, -1, :]).type(torch.LongTensor)]
            x = x + node_emb + hour_embed + day_embed
        else:
            x = x + node_emb
        x = self.padding_patch_layer(x)  # [B C (T+padding)]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [B C 12 16]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # 这一步是实现通道独立机制

        x = self.value_embedding(x)

        return self.dropout(x), N
