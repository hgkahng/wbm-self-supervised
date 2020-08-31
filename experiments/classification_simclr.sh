#!/bin/bash

read -p "Select data [wm811k, cifar10]: " DATA

if [[ $DATA == "wm811k" ]]; then
    INPUT_SIZE=64
    EPOCHS=100
    BATCH_SIZE=256
    OPTIMIZER="adamw"
    LEARNING_RATE=(0.01)
    WEIGHT_DECAY=(0.001)
    SCHEDULER="none"
    WARMUP_STEPS=0
elif [[ $DATA == "cifar10" ]]; then
    INPUT_SIZE=32
    EPOCHS=100
    BATCH_SIZE=256
    OPTIMIZER="adamw"
    LEARNING_RATE=(0.001)
    WEIGHT_DECAY=(0.01)
    SCHEDULER="none"
    WARMUP_STEPS=0
fi

BACKBONE_TYPE="resnet"
BACKBONE_CONFIG="18.original"
PROJECTOR_TYPE="mlp"
PROJECTOR_SIZE=128
DROPOUT=0.0
SMOOTHING=0.0

read -p "Select GPU number [0, 1, 2, 3]: "  DEVICE
DEVICE="cuda:$DEVICE"
read -p "Number of CPU threads: " NUM_WORKERS

read -p "Select evaluation metric: [loss, accuracy, f1, auc]: " EVAL_METRIC

PRETRAINED_DIR="./checkpoints/${DATA}/simclr/${BACKBONE_TYPE}.${BACKBONE_CONFIG}/"
read -p "Top directory of pretrained checkpoints (defaults to '${PRETRAINED_DIR}'): " TEMP_DIR
if [[ $TEMP_DIR == "" ]]; then
    echo "Finding models under '${PRETRAINED_DIR}' ..."
elif [ -d "$TEMP_DIR" ]; then
    echo "Valid directory provided: $TEMP_DIR"
    PRETRAINED_DIR=TEMP_DIR
elif [ ! -d "$TEMP_DIR" ]; then
    echo "Error: ${TEMP_DIR} not found. Terminating..."
    exit 1
fi

read -p "Enter a integer random seed: " SEED
cat "./experiments/classification_simclr.sh"
echo -e "\n"

for LR in $LEARNING_RATE; do
    for WD in $WEIGHT_DECAY; do
        for LP in 1.0; do
            echo -e "\n\n"
            echo "Dataset: ${DATA} "
            echo "Labeled Proportion: ${LP} "
            echo "Optimization: ${OPTIMIZER} (LR=${LR}, WD=${WD}) | Scheduler: ${SCHEDULER}"
            echo "Backbone: ${BACKBONE_TYPE}.${BACKBONE_CONFIG} "
            echo "Best models selected by: ${EVAL_METRIC} "
            echo "Random seed: ${SEED} "
            echo "GPU: ${DEVICE} | CPU threads: ${NUM_WORKERS}"
            python run_classification.py \
                --seed $SEED \
                --data $DATA \
                --input_size $INPUT_SIZE \
                --backbone_type $BACKBONE_TYPE \
                --backbone_config $BACKBONE_CONFIG \
                --label_proportion $LP \
                --epochs $EPOCHS \
                --batch_size $BATCH_SIZE \
                --num_workers $NUM_WORKERS \
                --device $DEVICE \
                --dropout $DROPOUT \
                --smoothing $SMOOTHING \
                --optimizer $OPTIMIZER \
                --learning_rate $LR \
                --weight_decay $WD \
                --scheduler $SCHEDULER \
                --warmup_steps $WARMUP_STEPS \
                --eval_metric $EVAL_METRIC \
                --freeze \
                --write_summary \
                --checkpoint_root "./checkpoints/" \
                simclr \
                --root $PRETRAINED_DIR \
                --projector_type $PROJECTOR_TYPE \
                --projector_size $PROJECTOR_SIZE
        done
    done
done
