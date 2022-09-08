#!/bin/sh
node_ip="$(ifconfig | grep "inet" | awk 'NR==1{print $2}')"

python -m debugpy \
    --listen $node_ip:1400 plot_training_results.py \
    --training_run=00004-ffhq-FFHQ_512_6_balanced-gpus2-batch8-gamma5 \
    --id=False