#!/bin/bash

read -p "Select data [wm811k, cifar10]: " DATA

if [ $DATA == "wm811k" ]; then
    INPUT_SIZE=64
    EPOCHS=500
    BATCH_SIZE=2048
    OPTIMIZER="sgd"
    LEARNING_RATE=(0.1)
    WEIGHT_DECAY=(0.001)
    SCHEDULER="cosine"
    WARMUP_STEPS=10
    TEMPERATURE=0.07
elif [ $DATA == "cifar10" ]; then
    INPUT_SIZE=32
    EPOCHS=1000
    BATCH_SIZE=1024
    OPTIMIZER="adamw"
    LEARNING_RATE=(0.01 0.001)
    WEIGHT_DECAY=(0.01 0.001)
    SCHEDULER="none"
    WARMUP_STEPS=0
    TEMPERATURE=0.07
fi

BACKBONE_TYPE="resnet"
BACKBONE_CONFIG="18.original"
PROJECTOR_TYPE="mlp"
PROJECTOR_SIZE=128

read -p "Select GPU number [0, 1, 2, 3]: " DEVICE
DEVICE="cuda:${DEVICE}"
read -p "Number of CPU threads: " NUM_WORKERS

cat "./experiments/pretext_simclr.sh"

for LR in $LEARNING_RATE; do
    for WD in $WEIGHT_DECAY; do
        echo -e "\n\n"
        echo "Pretraining with SimCLR, data: $DATA, GPU: $DEVICE."
        echo "Backbone: ${BACKBONE_TYPE}.${BACKBONE_CONFIG}"
        echo "Projector type: ${PROJECTOR_TYPE}"
        echo "Projector size: ${PROJECTOR_SIZE}"
        python run_simclr.py \
            --data $DATA \
            --input_size $INPUT_SIZE \
            --backbone_type $BACKBONE_TYPE \
            --backbone_config $BACKBONE_CONFIG \
            --projector_type $PROJECTOR_TYPE \
            --projector_size $PROJECTOR_SIZE \
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
            --write_summary
    done
done
echo "Finished."
