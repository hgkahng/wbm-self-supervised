#!/bin/bash

echo "Classification on Wafer 40, backbone trained with bigan."
echo "Different proportions of labeled training data will be used."

cat './experiments/classification_bigan.sh'

BACKBONE_TYPE="vgg"
BACKBONE_CONFIG="3"

for DATA_INDEX in {0..9}; do
    for LABELED in 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
        echo "Backbone: ${BACKBONE_TYPE}.${BACKBONE_CONFIG}"
        echo "Data Index: ${DATA_INDEX} | Labeled Proportion: ${LABELED}"
        python run_classification.py \
            --data_index $DATA_INDEX \
            --backbone_type $BACKBONE_TYPE \
            --backbone_config $BACKBONE_CONFIG \
            --in_channels 2 \
            --labeled $LABELED \
            --epochs 1000 \
            --batch_size 128 \
            --num_workers 1 \
            --device "cuda:0" \
            --smoothing 0.1 \
            --optimizer_type "adamw" \
            --learning_rate 0.001 \
            --weight_decay 0.01 \
            --beta1 0.9 \
            --checkpoint_root "./checkpoints/" \
            --write_summary \
            --pretrained_root "./checkpoints/bigan/" \
            --pretext "bigan" \
            --gan_type "dcgan" \
            --gan_config "3" \
            --freeze "block1 block2 block3"
    done
done

echo "Finished."
