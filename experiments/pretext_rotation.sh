#!/bin/bash

cat './experiments/pretext_rotation.sh'
echo "Pretraining with rotation, data: WM811k."

BACKBONE_TYPE="vgg"

for BACKBONE_CONFIG in "3" "6"; do
    echo "Backbone: ${BACKBONE_TYPE}.${BACKBONE_CONFIG}"
    python run_rotation.py \
        --backbone_type $BACKBONE_TYPE \
        --backbone_config $BACKBONE_CONFIG \
        --in_channels 2 \
        --epochs 100 \
        --batch_size 1024 \
        --num_workers 1 \
        --device "cuda:1" \
        --optimizer_type "adamw" \
        --learning_rate 0.001 \
        --weight_decay 0.01 \
        --beta1 0.9 \
        --checkpoint_root "./checkpoints/" \
        --write_summary \
        --save_every
done

echo "Finished."
