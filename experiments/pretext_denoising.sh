#!/bin/bash

echo "Pretraining with denoising autoencoder, data: WM811k."

read -p "Select CNN backbone [alexnet, vggnet, resnet]: " BACKBONE_TYPE
read -p "Specify backbone configuration: " BACKBONE_CONFIG

read -p "Augmentation [rotate, crop, shear, shift, noise, test]: " AUGMENTATION
if [ $AUGMENTATION != "test" ]; then
    read -p "Noise rate [0.00, 0.01, 0.05, 0.10]: " NOISE
else
    NOISE=0.00
fi

read -p "Optimizer: " OPTIMIZER
read -p "Learning rate: " LR
read -p "Weight decay: " WD
read -p "Epochs: " EPOCHS
read -p "Batch size: " BATCH_SIZE

read -p "Scheduler [cosine, none]: " SCHEDULER
if [ $SCHEDULER == "cosine" ]; then
    read -p "Linear warmup steps: " WARMUP_STEPS
elif [ $SCHEDULER == "restart" ]; then
    read -p "Warmup steps after restart: " WARMUP_STEPS
else
    WARMUP_STEPS=0
fi

read -p "Select GPU number [0, 1, 2, 3]: " DEVICE
read -p "Number of CPU threads: " NUM_WORKERS
DEVICE="cuda:${DEVICE}"

cat "./experiments/pretext_denoising.sh"
echo -e "\n\n"

python run_denoising.py \
    --data "wm811k" \
    --input_size 96 \
    --backbone_type $BACKBONE_TYPE \
    --backbone_config $BACKBONE_CONFIG \
    --augmentation $AUGMENTATION \
    --noise $NOISE \
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
