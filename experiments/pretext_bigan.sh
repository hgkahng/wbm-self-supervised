#!/bin/bash

cat './experiments/pretext_bigan.sh'
echo "Pretraining with bigan, data: WM811k."

BACKBONE_TYPE="vgg"

for BACKBONE_CONFIG in "3" "6"; do
    for GAN_TYPE in 'dcgan' 'vgg'; do
    echo "Backbone: ${BACKBONE_TYPE}.${BACKBONE_CONFIG}"
    echo "GAN: ${GAN_TYPE}"
    python run_bigan.py \
        --backbone_type $BACKBONE_TYPE \
        --backbone_config $BACKBONE_CONFIG \
        --in_channels 2 \
        --gan_type $GAN_TYPE \
        --gan_config $BACKBONE_CONFIG \
        --latent_size 50 \
        --epochs 100 \
        --batch_size 1024 \
        --num_workers 1 \
        --device "cuda:1" \
        --optimizer_type "adamw" \
        --learning_rate 0.001 \
        --weight_decay 0.01 \
        --beta1 0.5 \
        --checkpoint_root "./checkpoints/" \
        --write_summary \
        --save_every
    done
done

echo "Finished."
