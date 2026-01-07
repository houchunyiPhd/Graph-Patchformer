import torch
import torch.nn as nn
from layers.STGraphformer_Embed import DataEmbedding_inverted_patching
from layers.STGraphformer_Enc import Encoder, EncoderLayer, PgcnLayer
from layers.STGraphformer_Enc import FullAttention, AttentionLayer, Gcnlayer, FullGNN, nconv


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout()

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        return x

class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        padding = stride

        # patching and embedding
        self.patch_embedding = DataEmbedding_inverted_patching(configs.enc_in, configs.seq_len, configs.d_model,
                                                               patch_len, stride, padding, configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],

            [
                PgcnLayer(
                    Gcnlayer(
                        FullGNN(configs.k_hop, configs.d_ff),
                                configs.d_model,
                                configs.c_out,
                                configs.node_dim),
                    configs.d_model,
                    configs.dropout,
                    configs.batch_size,
                    configs.c_out
                ) for l in range(configs.e_layers)
            ],
        norm_layer = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2))
        )

        self.head_nf = configs.d_model * \
                       int((configs.seq_len - patch_len) / stride + 2)

        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 反归一化
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        enc_out, N = self.patch_embedding(x_enc, x_mark_enc)

        enc_out, adj_list = self.encoder(enc_out)


        enc_out = torch.reshape(
            enc_out, (-1, N, enc_out.shape[-2], enc_out.shape[-1]))

        enc_out = enc_out.permute(0, 1, 3, 2)
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]

