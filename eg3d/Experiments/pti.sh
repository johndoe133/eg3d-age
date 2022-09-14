#!/bin/sh
module load gcc/9.2.0
module load cuda/11.1

node_ip="$(ifconfig | grep "inet" | awk 'NR==1{print $2}')"

python -m debugpy \
    --listen $node_ip:1400 PTI/pti_pipeline.py --age 35 --image_name crop --preprocess False --model_path \
     ./training-runs/00118-ffhq-FFHQ_512-gpus2-batch8-gamma5/network-snapshot-000200.pkl \
    --w_iterations 500 --pti_iterations 500 --run_pti_inversion True