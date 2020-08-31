#!/bin/bash

echo "Pretraining with SemiCLR, data: WM811k."

cat "./experiments/pretext_semiclr.sh"

BACKBONE_TYPE="resnet"
BACKBONE_CONFIG="18.original"
INPUT_SIZE=56
PROJECTOR_SIZE=128

UNLABELED_BATCH_SIZE=4096
LABELED_BATCH_SIZE=1024

for PROJECTOR_TYPE in "mlp"; do
    echo "Backbone: ${BACKBONE_TYPE}.${BACKBONE_CONFIG}"
    echo "Projector type: ${PROJECTOR_TYPE}"
    echo "Projector size: ${PROJECTOR_SIZE}"
    python run_semiclr.py \
        --input_size 56 \
        --backbone_type $BACKBONE_TYPE \
        --backbone_config $BACKBONE_CONFIG \
        --projector_type $PROJECTOR_TYPE \
        --projector_size $PROJECTOR_SIZE \
        --temperature 0.07 \
        --label_proportion 1.0 \
        --epochs 100 \
        --unlabeled_batch_size $UNLABELED_BATCH_SIZE \
        --labeled_batch_size $LABELED_BATCH_SIZE \
        --num_workers 8 \
        --device "cuda:2" \
        --optimizer "sgd" \
        --learning_rate 1.0 \
         --weight_decay 0.001 \
        --scheduler "cosine" \
        --warmup_steps 10 \
        --checkpoint_root "./checkpoints/" \
        --write_summary
done
echo "Finished."
