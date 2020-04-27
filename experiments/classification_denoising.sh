#!/bin/bash

echo "Classification on Wafer 40, with backbone trained with denoising."
echo "Different proportions of labeled training data will be used."

cat "./experiments/classification_denoising.sh"

BACKBONE_TYPE="vgg"
BACKBONE_CONFIG="6a"

for DATA_INDEX in {0..29}; do
    for LABELED in 0.01 0.05 0.10 0.25 0.50 0.75 1.0; do
        echo "Backbone: ${BACKBONE_TYPE}.${BACKBONE_CONFIG}"
        echo "Data Index: ${DATA_INDEX} | Labeled Proportion: ${LABELED}"
        python run_classification.py \
            --data_index $DATA_INDEX \
            --backbone_type $BACKBONE_TYPE \
            --backbone_config $BACKBONE_CONFIG \
            --in_channels 2 \
            --labeled $LABELED \
            --epochs 3000 --batch_size 64 \
            --num_workers 0 --device "cuda:3" \
            --batch_norm --dropout 0.5 --smoothing 0.1 \
            --optimizer "adamw" --learning_rate 0.0001 --weight_decay 0.01 \
            --blockwise_learning_rates "block1=0.000001" "block2=0.00001" "block3=0.0001" \
            --checkpoint_root "./checkpointsNew/" \
            --write_summary \
            denoising \
            --root "./checkpoints/denoising/" \
            --noise 0.25
    done
done

echo "Finished."
