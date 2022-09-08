#!/bin/sh
module load gcc/9.2.0
module load cuda/11.1

node_ip="$(ifconfig | grep "inet" | awk 'NR==1{print $2}')"

python -m debugpy \
    --listen $node_ip:1400 PTI/visualize_pti.py \