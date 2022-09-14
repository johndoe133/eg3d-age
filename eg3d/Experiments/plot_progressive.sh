#!/bin/sh
module load gcc/9.2.0
module load cuda/11.1

node_ip="$(ifconfig | grep "inet" | awk 'NR==1{print $2}')"

export CUDA_VISIBLE_DEVICES=0,1

python -m debugpy \
    --listen $node_ip:1222 plot_progressive.py --network=./training-runs/00002-ffhq-FFHQ_512_6_balanced-gpus2-batch8-gamma5/network-snapshot-000200.pkl --seed=42 --outdir=out