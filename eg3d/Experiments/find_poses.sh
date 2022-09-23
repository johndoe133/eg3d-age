#!/bin/sh
module load gcc/9.2.0
module load cuda/11.1

export CUDA_VISIBLE_DEVICES=0,1

node_ip="$(ifconfig | grep "inet" | awk 'NR==1{print $2}')"

cd $HOME/Documents/eg3d-age/dataset_preprocessing/ffhq

python batch_mtcnn.py --in_root $HOME/Documents/eg3d-age/dataset_preprocessing/ffhq/Deep3DFaceRecon_pytorch/test

cd Deep3DFaceRecon_pytorch

python -m debugpy \
    --listen $node_ip:1500 test.py \
    --img_folder=$HOME/Documents/eg3d-age/dataset_preprocessing/ffhq/Deep3DFaceRecon_pytorch/test --gpu_ids=0 --name=pretrained --epoch=20

cd ..

python 3dface2idr_mat.py --in_root Deep3DFaceRecon_pytorch/checkpoints/pretrained/results/test/epoch_20_000000

python preprocess_cameras.py --source Deep3DFaceRecon_pytorch/checkpoints/pretrained/results/test/epoch_20_000000 --mode orig

python crop_images_in_the_wild.py --indir=$HOME/Documents/eg3d-age/dataset_preprocessing/ffhq/Deep3DFaceRecon_pytorch/test

cd ../../eg3d