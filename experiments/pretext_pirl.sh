#!/bin/bash

read -p "Select data [wm811k, cifar10]: " DATA

if [ $DATA == "wm811k" ]; then
    INPUT_SIZE=96
    EPOCHS=100
    BATCH_SIZE=256
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
    OPTIMIZER="sgd"
    LEARNING_RATE=(1e-2)
    WEIGHT_DECAY=(1e-3)
    SCHEDULER="cosine"
    WARMUP_STEPS=5
    TEMPERATURE=0.07
fi

read -p "Select CNN backbone [alexnet, vggnet, resnet]: " BACKBONE_TYPE
read -p "Specify backbone configuration: " BACKBONE_CONFIG

PROJECTOR_TYPE="mlp"
PROJECTOR_SIZE=128
LOSS_WEIGHT=0.5
TEMPERATURE=0.07
NUM_NEGATIVES=5000

read -p "Augmentation [rotate+crop, rotate, crop, cutout, shift, noise, test]: " AUGMENTATION
read -p "Select GPU number [0, 1, 2, 3]: " GPU_NUM
read -p "Number of CPU threads: " NUM_WORKERS
DEVICE="cuda:${GPU_NUM}"

cat "./experiments/pretext_pirl.sh"

for LR in "${LEARNING_RATE[@]}"
do
    for WD in "${WEIGHT_DECAY[@]}"
    do
        echo -e "\n"
        
        echo "Task: PIRL "
        echo "Data: ${DATA} (${INPUT_SIZE} x ${INPUT_SIZE})"
        
        echo "Backbone: ${BACKBONE_TYPE}.${BACKBONE_CONFIG}"
        echo "Head: ${PROJECTOR_TYPE} (dim=${PROJECTOR_SIZE})"
        
        echo "Optimizer: ${OPTIMIZER} (lr=${LR}, weight decay=${WD})"
        echo "Scheduler: ${SCHEDULER} (warmup=$WARMUP_STEPS)"
        
        echo "Epochs: ${EPOCHS} (batch size: ${BATCH_SIZE})"

        echo "Loss weight: ${LOSS_WEIGHT}"
        echo "Temperature: ${TEMPERATURE}"
        echo "Negative examples: ${NUM_NEGATIVES}"
        
        echo "Augmentation: ${AUGMENTATION}"
        echo "GPU: ${GPU_NUM} (${NUM_WORKERS} CPU threads)"

        python run_pirl.py \
            --data $DATA \
            --input_size $INPUT_SIZE \
            --backbone_type $BACKBONE_TYPE \
            --backbone_config $BACKBONE_CONFIG \
            --projector_type $PROJECTOR_TYPE \
            --projector_size $PROJECTOR_SIZE \
            --num_negatives $NUM_NEGATIVES \
            --loss_weight $LOSS_WEIGHT \
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
            --save_every 10 \
            --write_summary
    done
done
