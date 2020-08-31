#!/bin/bash

read -p "Select data [wm811k, cifar10]: " DATA

if [[ $DATA == "wm811k" ]]; then
    INPUT_SIZE=96
    EPOCHS=100
    BATCH_SIZE=256
    OPTIMIZER="sgd"
    LEARNING_RATE=(1e-2)
    WEIGHT_DECAY=(1e-3)
    SCHEDULER="cosine"
    WARMUP_STEPS=0
elif [[ $DATA == "cifar10" ]]; then
    INPUT_SIZE=32
    EPOCHS=100
    BATCH_SIZE=256
    OPTIMIZER="sgd"
    LEARNING_RATE=(1e-1)
    WEIGHT_DECAY=(1e-3)
    SCHEDULER="cosine"
    WARMUP_STEPS=0
fi

DROPOUT=0.5
SMOOTHING=0.1

read -p "Select CNN backbone [alexnet, vggnet, resnet]: " BACKBONE_TYPE
read -p "Specify backbone configuration: " BACKBONE_CONFIG
read -p "Augmentation [rotate, crop, shear, shift, noise]: " AUGMENTATION
read -p "Noise rate [0.01 0.05 0.10 0.25]: " NOISE
read -p "Select GPU number [0, 1, 2, 3]: " GPU_NUM
read -p "Number of CPU threads: " NUM_WORKERS
read -p "Select evaluation metric: [loss, accuracy, f1]: " EVAL_METRIC

DEVICE="cuda:$GPU_NUM"
SEED=(1 2 3 4)
for SD in "${SEED[@]}"
do
    for LR in "${LEARNING_RATE[@]}"
    do
        for WD in "${WEIGHT_DECAY[@]}"
        do
            for LP in 0.01 0.05 0.10 0.25 0.50 1.00;
            do
                echo -e "\n\n"
                echo "Random seed: $SD "
                echo "Dataset: $DATA (Labels: ${LP}) "
                echo "Backbone: ${BACKBONE_TYPE}.${BACKBONE_CONFIG} "
                echo "Projector: ${PROJECTOR_TYPE} (size=${PROJECTOR_SIZE})"
                echo "Optimization: ${OPTIMIZER} (LR=${LR}, WD=${WD})"
                echo "Scheduler: ${SCHEDULER} (warmup=${WARMUP_STEPS})"
                echo "Epochs: ${EPOCHS} (batch size: ${BATCH_SIZE})"
                echo "GPU: $DEVICE | CPU threads: ${NUM_WORKERS}"
                python run_classification.py \
                    --seed $SD \
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
                    --balance \
                    --augmentation $AUGMENTATION \
                    --optimizer $OPTIMIZER \
                    --learning_rate $LR \
                    --weight_decay $WD \
                    --scheduler $SCHEDULER \
                    --warmup_steps $WARMUP_STEPS \
                    --eval_metric $EVAL_METRIC \
                    --checkpoint_root "./checkpoints.temp/" \
                    --write_summary \
                    denoising \
                    --root "./checkpoints/wm811k/denoising/" \
                    --noise $NOISE
            done
        done
    done
done
