#!/bin/sh
module load gcc/9.2.0
module load cuda/11.1

node_ip="$(ifconfig | grep "inet" | awk 'NR==1{print $2}')"

export CUDA_VISIBLE_DEVICES=0,1

python -m debugpy \
    --listen $node_ip:1400 plot_progressive.py \
    --seed=42 \
    --network_folder=./training-runs/00004-ffhq-FFHQ_512_6_balanced-gpus2-batch8-gamma5