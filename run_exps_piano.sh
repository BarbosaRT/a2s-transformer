#!/bin/bash

for tokenization in kern st_plus midi2score; do
    python -u train.py \
        --ds_name asap-piano-excerpts \
        --model_type transformer \
        --audio_mode spectrogram \
        --tokenization "$tokenization" \
        --batch_size 20 \
        --patience 10 \
        --attn_window 100 \
        --epochs 150 \
        --use_checkpoint False \
        --check_val_every_n_epoch 50 \
        --limit_val_batches 64
    python -u test.py --ds_name asap-piano-excerpts --model_type transformer --audio_mode spectrogram --tokenization "$tokenization" --checkpoint_path weights/transformer/asap-piano-excerpts$([ "$tokenization" == "kern" ] || echo "_$tokenization").ckpt
done
