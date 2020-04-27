#!/bin/bash

echo "Classification on Wafer 40, backbone trained with pirl."
echo "Different proportions of labeled training data will be used."

cat './experiments/classification_pirl.sh'

BACKBONE_TYPE="resnet"
BACKBONE_CONFIG="18.original"

for NOISE in 0.05; do
    for NUM_NEGATIVES in 5000; do
        for DATA_INDEX in {0..9}; do
            for LABELED in 0.001 0.005 0.01 0.05 0.10 0.25 0.50 1.0; do
                echo "Backbone: ${BACKBONE_TYPE}.${BACKBONE_CONFIG}"
                echo "Data Index: ${DATA_INDEX} | Labeled Proportion: ${LABELED}"
                python run_classification.py \
                    --data_index $DATA_INDEX \
                    --backbone_type $BACKBONE_TYPE --backbone_config $BACKBONE_CONFIG \
                    --labeled $LABELED \
                    --epochs 100 --batch_size 256 \
                    --num_workers 8 --device "cuda:0" \
                    --dropout 0.5 --smoothing 0.1 \
                    --optimizer "adamw" --learning_rate 0.001 --weight_decay 0.01 \
                    --eval_metric "f1" \
                    --checkpoint_root "./checkpoints/" \
                    --write_summary \
                    pirl \
                    --root "./checkpoints/pirl/" \
                    --model_type "last" \
                    --projector_type "linear" --projector_size 128 \
                    --num_negatives $NUM_NEGATIVES \
                    --noise $NOISE --rotate
            done
        done
    done
done
echo "Finished."