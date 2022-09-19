#!/bin/sh
module load gcc/9.2.0
module load cuda/11.1

node_ip="$(ifconfig | grep "inet" | awk 'NR==1{print $2}')"

export CUDA_VISIBLE_DEVICES=0,1

python -m debugpy \
    --listen $node_ip:1500 evaluation.py \
    --network_folder=./training-runs/00122-ffhq-FFHQ_512-gpus2-batch8-gamma5 \
    --run_eval False