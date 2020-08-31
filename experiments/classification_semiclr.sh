#!/bin/bash

echo "Classification on Wafer 40, backbone trained with pirl."
echo "Different proportions of labeled training data will be used."

cat "./experiments/classification_semiclr.sh"

BACKBONE_TYPE="resnet"
BACKBONE_CONFIG="18.original"
PROJECTOR_TYPE="mlp"
PRETRAINED_MODEL="best"
INPUT_SIZE=56

for DATA_INDEX in {0..9}; do
    for LABELED in 1.0; do
        echo "Backbone: ${BACKBONE_TYPE}.${BACKBONE_CONFIG}"
        echo "Data Index: ${DATA_INDEX} | Labeled Proportion: ${LABELED}"
        python run_classification.py \
            --data_index $DATA_INDEX \
            --input_size $INPUT_SIZE \
            --backbone_type $BACKBONE_TYPE --backbone_config $BACKBONE_CONFIG \
            --labeled $LABELED \
            --epochs 100 --batch_size 512 \
            --num_workers 8 --device "cuda:3" \
            --dropout 0.5 --smoothing 0.1 \
            --optimizer "adamw" --learning_rate 0.001 --weight_decay 0.01 \
            --eval_metric "f1" \
            --checkpoint_root "./checkpoints/" \
            --write_summary \
            semiclr \
            --root "./checkpoints/semiclr/resnet.18.original/2020-06-15_14:10:37/" \
            --model_type $PRETRAINED_MODEL \
            --projector_type $PROJECTOR_TYPE --projector_size 128 \
            --label_proportion 1.0
    done
done
echo "Finished."