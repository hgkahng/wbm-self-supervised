#!/bin/bash

cat "./experiments/pretext_pirl.sh"
echo "Pretraining with pirl, data: WM811k."

BACKBONE_TYPE="resnet"
BACKBONE_CONFIG="18.original"
PROJECTOR_SIZE=128

for PROJECTOR_TYPE in "mlp" "linear"; do
    for NUM_NEGATIVES in 10000; do
        for NOISE_PROB in 0.00 0.05 0.10; do
            echo "Backbone: ${BACKBONE_TYPE}.${BACKBONE_CONFIG}"
            echo "Projector type: ${PROJECTOR_TYPE}"
            echo "Projector size: ${PROJECTOR_SIZE}"
            echo "Negative samples: ${NUM_NEGATIVES}"
            echo "Noise: Bernoulli (${NOISE_PROB})"
            python run_pirl.py \
                --backbone_type $BACKBONE_TYPE --backbone_config $BACKBONE_CONFIG \
                --projector_type $PROJECTOR_TYPE --projector_size $PROJECTOR_SIZE \
                --num_negatives $NUM_NEGATIVES \
                --loss_weight 0.9 --temperature 0.07 \
                --rotate --noise $NOISE_PROB \
                --epochs 150 --batch_size 512 \
                --num_workers 8 --device "cuda:3" \
                --optimizer "adamw" --learning_rate 0.0001 --weight_decay 0.0005 \
                --checkpoint_root "./checkpoints/" --write_summary --save_every
        done
    done
done
echo "Finished."
