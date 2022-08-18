#!/bin/sh
module load gcc/9.2.0
module load cuda/11.1

node_ip="$(ifconfig | grep "inet" | awk 'NR==1{print $2}')"

python -m debugpy \
    --listen $node_ip:1222 train.py \
    --outdir=./training-runs \
    --cfg=ffhq \
    --data=./datasets/FFHQ_128 \
    --gpus=1 \
    --batch=32 \
    --gamma=5 \
    --gen_pose_cond=True 