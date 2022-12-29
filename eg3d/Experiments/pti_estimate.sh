#!/bin/sh
module load gcc/9.2.0
module load cuda/11.3

node_ip="$(ifconfig | grep "inet" | awk 'NR==1{print $2}')"

export CUDA_VISIBLE_DEVICES=0,1

python -m debugpy \
    --listen $node_ip:1500 pti_estimation.py --model_path /zhome/d7/6/127158/Documents/eg3d-age/eg3d/training-runs/00179/00000-ffhq-FFHQ-gpus2-batch8-gamma5/network-snapshot-001200.pkl 