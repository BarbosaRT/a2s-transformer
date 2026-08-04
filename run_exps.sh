#!/bin/bash

MODELS="crnn transformer transformer2"

usage() {
    echo "Usage: $0 [--model crnn|transformer|transformer2]"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            if [ -z "$2" ]; then usage; fi
            MODELS="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

for model_type in $MODELS; do
    case $model_type in
        crnn)
            for train_ds in Quartets Beethoven Haydn Mozart; do
                python -u train.py --ds_name "$train_ds" --model_type "$model_type" --batch_size 1 --patience 5
                for test_ds in Quartets Beethoven Haydn Mozart; do
                    if [ "$train_ds" != "$test_ds" ]; then
                        python -u test.py --ds_name "$test_ds" --model_type "$model_type" --checkpoint_path weights/$model_type/$train_ds.ckpt
                    fi
                done
            done
            ;;
        transformer)
            for train_ds in Quartets Beethoven Haydn Mozart; do
                python -u train.py --ds_name "$train_ds" --model_type "$model_type" --batch_size 1 --patience 5 --attn_window 100
                for test_ds in Quartets Beethoven Haydn Mozart; do
                    if [ "$train_ds" != "$test_ds" ]; then
                        python -u test.py --ds_name "$test_ds" --model_type "$model_type" --checkpoint_path weights/$model_type/$train_ds.ckpt
                    fi
                done
            done
            for tokenization in kern st_plus midi2score; do
                python -u train.py --ds_name asap-piano-excerpts --model_type "$model_type" --audio_mode spectrogram --tokenization "$tokenization" --batch_size 1 --patience 5 --attn_window 100
                python -u test.py --ds_name asap-piano-excerpts --model_type "$model_type" --audio_mode spectrogram --tokenization "$tokenization" --checkpoint_path weights/$model_type/asap-piano-excerpts$([ "$tokenization" == "kern" ] || echo "_$tokenization").ckpt
            done
            ;;
        transformer2)
            for tokenization in kern st_plus midi2score; do
                python -u train.py --ds_name asap-piano-excerpts --model_type "$model_type" --audio_mode musicfm --tokenization "$tokenization" --batch_size 1 --patience 5 --attn_window 100 --freeze_encoder --musicfm_path musicfm/data/pretrained_fma.pt --musicfm_stat_path musicfm/data/fma_stats.json
                python -u test.py --ds_name asap-piano-excerpts --model_type "$model_type" --audio_mode musicfm --tokenization "$tokenization" --checkpoint_path weights/$model_type/asap-piano-excerpts$([ "$tokenization" == "kern" ] || echo "_$tokenization").ckpt
            done
            ;;
        *)
            echo "Unknown model type: $model_type"
            usage
            ;;
    esac
done
