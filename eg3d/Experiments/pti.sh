#!/bin/sh
module load gcc/9.2.0
module load cuda/11.1

node_ip="$(ifconfig | grep "inet" | awk 'NR==1{print $2}')"

export CUDA_VISIBLE_DEVICES=1

python -m debugpy \
    --listen $node_ip:1500 PTI/pti_pipeline.py --image_name 007606 --preprocess False \
    --model_path ./training-runs/00187/00000-ffhq-FFHQ-gpus2-batch8-gamma5/network-snapshot-001440.pkl\
    --w_iterations 500 --pti_iterations 1000 --run_pti_inversion False --trunc 0.75

