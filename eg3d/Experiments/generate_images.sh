#!/bin/sh
module load gcc/9.2.0
module load cuda/11.3

node_ip="$(ifconfig | grep "inet" | awk 'NR==1{print $2}')"

export CUDA_VISIBLE_DEVICES=0,1

python -m debugpy \
    --listen $node_ip:1500 plot_progressive.py \
    --seed=7652 \
    --network_folder=./training-runs/00187/00000-ffhq-FFHQ-gpus2-batch8-gamma5 \
    --trunc=0.75 --calibrated=True\
