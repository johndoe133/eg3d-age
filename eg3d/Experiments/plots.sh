#!/bin/sh
node_ip="$(ifconfig | grep "inet" | awk 'NR==1{print $2}')"

python -m debugpy \
    --listen $node_ip:1500 plot_training_results.py \
    --training_run=00191\
    --id=True --comb_dir=True