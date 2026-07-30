import math

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torchinfo import summary
from lightning.pytorch import LightningModule

from musicfm.model.musicfm_25hz import MusicFM25Hz
from networks.transformer2.decoder import Decoder
from my_utils.metrics import compute_metrics
from my_utils.ar_dataset import SOS_TOKEN, EOS_TOKEN


SAMPLE_RATE = 24000
HOP_LENGTH = 240
SUBSAMPLING = 4
ENC_FRAME_RATE = SAMPLE_RATE // (HOP_LENGTH * SUBSAMPLING)  # 25 Hz
ENC_FRAMES_PER_SAMPLE = HOP_LENGTH * SUBSAMPLING  # 960


class A2STransformer(LightningModule):
    def __init__(
        self,
        max_seq_len,
        max_audio_len,
        w2i,
        i2w,
        ytest_i2w=None,
        attn_window=-1,
        teacher_forcing_prob=0.5,
        is_flash=False,
        musicfm_path=None,
        musicfm_stat_path=None,
        musicfm_layer_ix=12,
        freeze_encoder=False,
    ):
        super(A2STransformer, self).__init__()
        self.save_hyperparameters()
        self.w2i = w2i
        self.i2w = i2w
        self.ytest_i2w = ytest_i2w if ytest_i2w is not None else i2w
        self.padding_idx = w2i["<PAD>"]
        self.max_audio_len = max_audio_len
        self.max_seq_len = max_seq_len
        self.teacher_forcing_prob = teacher_forcing_prob
        self.freeze_encoder = freeze_encoder
        self.musicfm_layer_ix = musicfm_layer_ix

        self.encoder = MusicFM25Hz(
            is_flash=is_flash,
            stat_path=musicfm_stat_path,
            model_path=musicfm_path,
        )
        self.encoder_proj = nn.Linear(1024, 256)

        self.decoder = Decoder(
            output_size=len(self.w2i),
            max_seq_len=self.max_seq_len,
            num_embeddings=len(self.w2i),
            padding_idx=self.padding_idx,
            attn_window=attn_window,
        )

        if freeze_encoder:
            self.encoder.requires_grad_(False)

        self.compute_loss = CrossEntropyLoss(ignore_index=self.padding_idx)
        self.Y = []
        self.YHat = []

    def summary(self):
        print("Encoder (MusicFM25Hz)")
        print("Decoder")
        tgt_size = [1, self.max_seq_len]
        memory_size = [1, self.max_audio_len // ENC_FRAMES_PER_SAMPLE, 256]
        memory_len_size = [1]
        try:
            summary(
                self.decoder,
                input_size=[tgt_size, memory_size, memory_len_size],
                dtypes=[torch.int64, torch.float32, torch.int64],
            )
        except Exception:
            pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=2e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=150,
            eta_min=2e-5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def forward(self, x, xl, y_in):
        if self.freeze_encoder:
            return self.decoder(tgt=y_in, memory=x, memory_len=xl)
        _, hidden_states = self.encoder.get_predictions(x)
        emb = hidden_states[self.musicfm_layer_ix]
        x = self.encoder_proj(emb)
        xl_new = torch.ceil(xl.float() / ENC_FRAMES_PER_SAMPLE).long()
        y_out_hat = self.decoder(tgt=y_in, memory=x, memory_len=xl_new)
        return y_out_hat

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def apply_teacher_forcing(self, y):
        y_errored = y.clone()
        random_mask = torch.rand_like(y_errored, dtype=torch.float) < self.teacher_forcing_prob
        non_padding_mask = y != self.padding_idx
        combined_mask = random_mask & non_padding_mask
        random_indices = torch.randint(0, len(self.w2i), y_errored.shape, device=y_errored.device)
        y_errored = torch.where(combined_mask, random_indices, y_errored)
        return y_errored

    def training_step(self, batch, batch_idx):
        x, xl, y_in, y_out = batch
        y_in = self.apply_teacher_forcing(y_in)
        yhat = self.forward(x=x, xl=xl, y_in=y_in)
        loss = self.compute_loss(yhat, y_out)
        self.log("train_loss", loss, prog_bar=True, logger=True, on_epoch=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, xl, y = batch
        B = x.size(0)
        device = x.device

        _, hidden_states = self.encoder.get_predictions(x)
        x = self.encoder_proj(hidden_states[self.musicfm_layer_ix])
        xl_new = torch.ceil(xl.float() / ENC_FRAMES_PER_SAMPLE).long()

        sos = self.w2i[SOS_TOKEN]
        eos = self.w2i[EOS_TOKEN]

        y_in = torch.full((B, 1), sos, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        decoded = [[] for _ in range(B)]

        for _ in range(self.max_seq_len):
            y_out_hat = self.decoder(tgt=y_in, memory=x, memory_len=xl_new)
            next_tok = y_out_hat[:, :, -1].argmax(dim=-1)

            for i in range(B):
                if not finished[i]:
                    tok_id = next_tok[i].item()
                    decoded[i].append(self.i2w[tok_id])
                    if tok_id == eos:
                        finished[i] = True

            if finished.all():
                break

            y_in = torch.cat([y_in, next_tok.unsqueeze(1)], dim=1)

        for i in range(B):
            y_true = [self.ytest_i2w[t.item()] for t in y[i][1:]]
            self.Y.append(y_true)
            self.YHat.append(decoded[i])

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    @torch.no_grad()
    def on_validation_epoch_start(self):
        self.Y.clear()
        self.YHat.clear()

    @torch.no_grad()
    def on_validation_epoch_end(self, name="val", print_random_samples=False):
        metrics = compute_metrics(y_true=self.Y, y_pred=self.YHat)
        for k, v in metrics.items():
            self.log(f"{name}_{k}", v, prog_bar=True, sync_dist=True, logger=True, on_epoch=True)
        if print_random_samples and len(self.Y) > 0:
            index = torch.randint(0, len(self.Y), (1,)).item()
            print(f"Ground truth - {self.Y[index]}")
            print(f"Prediction - {self.YHat[index]}")
        self.Y.clear()
        self.YHat.clear()
        return metrics

    @torch.no_grad()
    def on_test_epoch_end(self):
        return self.on_validation_epoch_end(name="test", print_random_samples=True)
