#!/bin/bash

read -p "Select data [wm811k, cifar10]: " DATA

if [ $DATA == "wm811k" ]; then
    INPUT_SIZE=96
    EPOCHS=1000
    BATCH_SIZE=1024
    OPTIMIZER="sgd"
    LEARNING_RATE=(1e-2)
    WEIGHT_DECAY=(1e-3)
    SCHEDULER="cosine"
    WARMUP_STEPS=5
    TEMPERATURE=0.07
elif [ $DATA == "cifar10" ]; then
    INPUT_SIZE=32
    EPOCHS=1000
    BATCH_SIZE=1024
    OPTIMIZER="adamw"
    LEARNING_RATE=(1e-3)
    WEIGHT_DECAY=(5e-4)
    SCHEDULER="none"
    WARMUP_STEPS=0
    TEMPERATURE=0.1
fi

read -p "Select CNN backbone [resnet]: " BACKBONE_TYPE
if [ $BACKBONE_TYPE == "resnet" ]; then
    read -p "Specify backbone configuration [XX.original, XX.wide, XX.tiny]: " BACKBONE_CONFIG
else
    echo "'${BACKBONE_TYPE}' not supported."
    exit 1
fi

PROJECTOR_TYPE="mlp"
PROJECTOR_SIZE=128

read -p "Augmentation: " AUGMENTATION
read -p "Select GPU number [0, 1, 2, 3]: " GPU_NUM
read -p "Number of CPU threads: " NUM_WORKERS
DEVICE="cuda:${GPU_NUM}"

for LR in "${LEARNING_RATE[@]}"; do
    for WD in "${WEIGHT_DECAY[@]}"; do
        
        echo -e "\n"

        echo "Task: SimCLR"
        echo "Data: ${DATA} (${INPUT_SIZE} x ${INPUT_SIZE})"

        echo "Backbone: ${BACKBONE_TYPE}.${BACKBONE_CONFIG}"
        echo "Head: ${PROJECTOR_TYPE} (dim=${PROJECTOR_SIZE})"

        echo "Optimizer: ${OPTIMIZER} (lr=${LR}, weight decay=${WD})"
        echo "Scheduler: ${SCHEDULER} (warmup=$WARMUP_STEPS)"

        echo "Epochs: ${EPOCHS} (batch size: ${BATCH_SIZE})"

        echo "Temperature: ${TEMPERATURE}"
        echo "Augmentation: ${AUGMENTATION}"
        echo "GPU: ${GPU_NUM} (${NUM_WORKERS} CPU threads)"

        python run_simclr.py \
            --data $DATA \
            --input_size $INPUT_SIZE \
            --backbone_type $BACKBONE_TYPE \
            --backbone_config $BACKBONE_CONFIG \
            --projector_type $PROJECTOR_TYPE \
            --projector_size $PROJECTOR_SIZE \
            --temperature $TEMPERATURE \
            --augmentation $AUGMENTATION \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE \
            --num_workers $NUM_WORKERS \
            --device $DEVICE \
            --optimizer $OPTIMIZER \
            --learning_rate $LR \
            --weight_decay $WD \
            --scheduler $SCHEDULER \
            --warmup_steps $WARMUP_STEPS \
            --checkpoint_root "./checkpoints/" \
            --save_every 100 \
            --write_summary
    done
done