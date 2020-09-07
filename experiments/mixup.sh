#!/bin/bash

python run_mixup.py \
    --seed 0 \
    --data "wm811k" \
    --input_size 64 \
    --backbone_type "alexnet" \
    --backbone_config "batch_norm" \
    --epochs 100 \
    --batch_size 256 \
    --num_workers 4 \
    --device "cuda:3" \
    --balance \
    --augmentation "test" \
    --optimizer "sgd" \
    --learning_rate 1e-2 \
    --weight_decay 1e-3 \
    --scheduler "cosine" \
    --warmup_steps 0 \
    --dropout 0.5 \
    --eval_metric "f1" \
    --checkpoint_root "./checkpoints/" \
    --write_summary
