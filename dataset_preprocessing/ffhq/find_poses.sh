#!/bin/sh
module load gcc/9.2.0
module load cuda/11.1

node_ip="$(ifconfig | grep "inet" | awk 'NR==1{print $2}')"

python batch_mtcnn.py --in_root /zhome/d1/9/127646/Documents/eg3d-age/dataset_preprocessing/ffhq/Deep3DFaceRecon_pytorch/test

cd Deep3DFaceRecon_pytorch

python -m debugpy \
    --listen $node_ip:1400 test.py \
    --img_folder=/zhome/d1/9/127646/Documents/eg3d-age/dataset_preprocessing/ffhq/Deep3DFaceRecon_pytorch/test --gpu_ids=0 --name=pretrained --epoch=20

cd ..

python 3dface2idr_mat.py --in_root Deep3DFaceRecon_pytorch/checkpoints/pretrained/results/test/epoch_20_000000

python preprocess_cameras.py --source Deep3DFaceRecon_pytorch/checkpoints/pretrained/results/test/epoch_20_000000 --mode orig

python crop_images_in_the_wild.py --indir=/zhome/d1/9/127646/Documents/eg3d-age/dataset_preprocessing/ffhq/Deep3DFaceRecon_pytorch/test