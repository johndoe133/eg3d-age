#!/bin/sh
module load gcc/9.2.0
module load cuda/11.1

node_ip="$(ifconfig | grep "inet" | awk 'NR==1{print $2}')"

export CUDA_VISIBLE_DEVICES=0,1

python -m debugpy \
    --listen $node_ip:1222 train.py \
    --outdir=./training-runs \
    --cfg=ffhq \
    --data=./datasets/FFHQ_512 \
    --gpus=4 \
    --batch=16 \
    --gamma=5 \
    --gen_pose_cond=True \
    --resume=networks/ffhqrebalanced512-64.pkl \
    --neural_rendering_resolution_initial=64 