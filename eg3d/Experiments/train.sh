#!/bin/sh
module load gcc/9.2.0
module load cuda/11.3

node_ip="$(ifconfig | grep "inet" | awk 'NR==1{print $2}')"

export CUDA_VISIBLE_DEVICES=1

python -m debugpy \
    --listen $node_ip:1500 train.py \
    --outdir=./training-runs \
    --cfg=ffhq \
    --data=/work3/morbj/FFHQ/ \
    --gpus=1 \
    --batch=4 \
    --gamma=5 \
    --gen_pose_cond=True \
    --age_scale=10 \
    --id_scale=10 \
    --age_loss_fn=CAT \
    --age_version=v2\
    --resume=/zhome/d7/6/127158/Documents/eg3d-age/eg3d/networks/ffhqrebalanced512-128.pkl \
    --neural_rendering_resolution_initial=128\
    --age_min=0\
    --age_max=75\
    --id_model=MagFace --metrics=None\
    --crop_before_estimate_ages=False\