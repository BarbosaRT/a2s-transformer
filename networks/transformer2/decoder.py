import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_func


class PositionalEncoding1D(nn.Module):
    def __init__(self, max_len, emb_dim, dropout_p: float = 0.1):
        super(PositionalEncoding1D, self).__init__()
        self.dropout = nn.Dropout(p=dropout_p)

        pos = torch.arange(max_len).unsqueeze(1)
        den = torch.pow(10000, torch.arange(0, emb_dim, 2) / emb_dim)

        pe = torch.zeros(1, max_len, emb_dim)
        pe[0, :, 0::2] = torch.sin(pos / den)
        pe[0, :, 1::2] = torch.cos(pos / den)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class WindowedCausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, nhead, attn_window=-1, dropout=0.0):
        super().__init__()
        self.nhead = nhead
        self.head_dim = embed_dim // nhead
        self.attn_window = attn_window
        self.dropout_p = dropout
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, L, C = x.shape
        qkv = self.qkv_proj(x).view(B, L, 3, self.nhead, self.head_dim)
        q, k, v = qkv.unbind(2)
        left = (self.attn_window - 1) if self.attn_window > 0 else -1
        out = flash_attn_func(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            causal=True,
            window_size=(left, 0),
        )
        return self.out_proj(out.reshape(B, L, C))


class FlashDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, attn_window=-1, dropout=0.1):
        super().__init__()
        self.self_attn = WindowedCausalSelfAttention(d_model, nhead, attn_window, dropout)
        self.cross_attn_q = nn.Linear(d_model, d_model)
        self.cross_attn_k = nn.Linear(d_model, d_model)
        self.cross_attn_v = nn.Linear(d_model, d_model)
        self.cross_attn_out = nn.Linear(d_model, d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.nhead = nhead
        self.head_dim = d_model // nhead

    def _sdpa_cross_attention(self, tgt, memory, memory_key_padding_mask):
        B, T, C = tgt.shape
        _, S, _ = memory.shape
        q = self.cross_attn_q(tgt).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = self.cross_attn_k(memory).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        v = self.cross_attn_v(memory).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        attn_mask = None
        if memory_key_padding_mask is not None:
            attn_mask = (memory_key_padding_mask < 0.5).unsqueeze(1).unsqueeze(2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.cross_attn_out(out)

    def forward(self, tgt, memory, memory_key_padding_mask=None):
        tgt2 = self.norm1(tgt)
        tgt2 = self.self_attn(tgt2)
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self._sdpa_cross_attention(tgt2, memory, memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class FlashDecoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, num_layers, attn_window):
        super().__init__()
        self.layers = nn.ModuleList([
            FlashDecoderLayer(d_model, nhead, dim_feedforward, attn_window, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, tgt, memory, memory_key_padding_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, memory_key_padding_mask)
        return tgt


class Decoder(nn.Module):
    def __init__(
        self,
        output_size: int,
        max_seq_len: int,
        num_embeddings: int,
        embedding_dim: int = 256,
        padding_idx: int = 0,
        ff_dim: int = 256,
        dropout_p: float = 0.1,
        nhead: int = 4,
        num_transformer_layers: int = 8,
        attn_window: int = -1,
    ):
        super(Decoder, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self.pos_1d = PositionalEncoding1D(
            max_len=max_seq_len,
            emb_dim=embedding_dim,
            dropout_p=dropout_p,
        )

        self.attn_window = attn_window
        self.flash_decoder = FlashDecoder(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout_p,
            num_layers=num_transformer_layers,
            attn_window=attn_window,
        )

        self.out_layer = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=output_size,
            kernel_size=1,
        )

    def forward(self, tgt, memory, memory_len):
        tgt_emb = self.pos_1d(self.embedding(tgt))

        memory_key_padding_mask = self.get_memory_key_padding_mask(memory, memory_len)

        tgt_pred = self.flash_decoder(
            tgt=tgt_emb,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        tgt_pred = tgt_pred.permute(0, 2, 1).contiguous()
        tgt_pred = self.out_layer(tgt_pred)

        return tgt_pred

    def get_memory_key_padding_mask(self, memory, memory_len):
        if memory_len is None:
            return None
        B, S = memory.shape[:2]
        positions = torch.arange(S, device=memory.device).unsqueeze(0).expand(B, S)
        return (positions >= memory_len.unsqueeze(1)).to(torch.float32)
