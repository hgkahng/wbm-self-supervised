#!/bin/bash

cat './experiments/pretext_jigsaw.sh'
echo "Pretraining with jigsaw, data: WM811k."

BACKBONE_TYPE="vgg"

for BACKBONE_CONFIG in "3"; do
    for NUM_PERMUTATIONS in 100; do
        echo "Backbone: ${BACKBONE_TYPE}.${BACKBONE_CONFIG}"
        echo "Permutations: ${NUM_PERMUTATIONS}"
        python run_jigsaw.py \
            --backbone_type $BACKBONE_TYPE \
            --backbone_config $BACKBONE_CONFIG \
            --in_channels 2 \
            --num_patches 9 \
            --num_permutations $NUM_PERMUTATIONS \
            --epochs 100 \
            --batch_size 1024 \
            --num_workers 1 \
            --device "cuda:1" \
            --smoothing 0.1 \
            --optimizer_type "adamw" \
            --learning_rate 0.001 \
            --weight_decay 0.01 \
            --beta1 0.9 \
            --checkpoint_root "./checkpoints/" \
            --write_summary \
            --save_every
    done
done

echo "Finished."
