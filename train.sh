#!/usr/bin/bash

run=`ls .log/  | wc -l`
echo "Writing to .log/training-$run.log"

WANDB_MODE=offline \
OMP_NUM_THREADS=200 \
CUDA_VISIBLE_DEVICES=3 \
nohup torchrun --standalone --nproc_per_node=gpu main.py \
    --model vit_base_patch16_224 \
    --auto_resume false \
    --drop_path 0.1 \
    --batch_size 512 \
    --lr 4e-3 \
    --update_freq 8 \
    --model_ema true \
    --model_ema_eval true \
    --data_path /local/scratch/b/mfdl/datasets/imagenet-1k/images/ \
    --output_dir models/dyt \
    --enable_wandb true \
    --log_dir tensorboard/ \
    --dynamic_tanh false > .log/training-$run.log 2>&1 &
