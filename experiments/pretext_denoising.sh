#!/bin/bash

cat "./experiments/pretext_denoising.sh"
echo "Pretraining with denoising, data: WM811k."

BACKBONE_TYPE="vgg"
BACKBONE_CONFIG="9a"

for NOISE_PROB in 0.10 0.25; do
    echo "Backbone: ${BACKBONE_TYPE}.${BACKBONE_CONFIG}"
    echo "Noise: Bernoulli (${NOISE_PROB})"
    python run_denoising.py \
        --backbone_type $BACKBONE_TYPE \
        --backbone_config $BACKBONE_CONFIG \
        --in_channels 2 \
        --noise $NOISE_PROB \
        --epochs 500 --batch_size 1024 \
        --num_workers 0 --device "cuda:3" \
        --batch_norm \
        --optimizer "adamw" --learning_rate 0.0005 --weight_decay 0.01 \
        --checkpoint_root "./checkpoints/" \
        --write_summary \
        --save_every
done
echo "Finished."
