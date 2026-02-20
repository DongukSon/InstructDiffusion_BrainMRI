#!/bin/bash
export CUDA_VISIBLE_DEVICES=4,5,6,7

LOG_FILE="logs/train_logs/train_v0128_1_cm+seg.log"
mkdir -p logs

echo "Starting training at $(date)" | tee $LOG_FILE

nohup torchrun --nproc_per_node=4 main.py \
  --name v0128_1_cm+seg \
  --base configs/brain_mri_finetune.yaml \
  --train \
  --logdir logs/instruct_diffusion \
  >> $LOG_FILE 2>&1 &

PID=$!
echo "Training started in background. PID: $PID"
echo "Log file: $LOG_FILE"
echo "Monitor: tail -f $LOG_FILE"