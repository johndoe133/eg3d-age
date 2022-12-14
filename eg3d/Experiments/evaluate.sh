#!/bin/sh
module load gcc/9.2.0
module load cuda/11.1

node_ip="$(ifconfig | grep "inet" | awk 'NR==1{print $2}')"

export CUDA_VISIBLE_DEVICES=0

python -m debugpy \
    --listen $node_ip:1500 Evaluation/run_evaluation.py \
    --network_folder=./training-runs/00187/00000-ffhq-FFHQ-gpus2-batch8-gamma5\
    --age_model_name v2 --run_generate_data True --generate_image_folder False \
    --trunc 0.75 --calibrated True
    
