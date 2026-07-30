import torch
import torch.nn as nn


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
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=embedding_dim,
                nhead=nhead,
                dim_feedforward=ff_dim,
                dropout=dropout_p,
                batch_first=True,
            ),
            num_layers=num_transformer_layers,
        )

        self.out_layer = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=output_size,
            kernel_size=1,
        )

    def forward(self, tgt, memory, memory_len):
        tgt_emb = self.pos_1d(self.embedding(tgt))

        memory_key_padding_mask = self.get_memory_key_padding_mask(memory, memory_len)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1), tgt.device)

        tgt_pred = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=None,
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
        return positions >= memory_len.unsqueeze(1)
