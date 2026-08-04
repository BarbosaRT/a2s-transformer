#!/bin/bash

for tokenization in kern st_plus midi2score; do
    python -u train.py --ds_name asap-piano-excerpts --model_type transformer --audio_mode spectrogram --tokenization "$tokenization" --batch_size 1 --patience 5 --attn_window 100
    python -u test.py --ds_name asap-piano-excerpts --model_type transformer --audio_mode spectrogram --tokenization "$tokenization" --checkpoint_path weights/transformer/asap-piano-excerpts$([ "$tokenization" == "kern" ] || echo "_$tokenization").ckpt
done
