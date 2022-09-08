import os
import sys
import pickle
import numpy as np
from PIL import Image
import torch
from configs import paths_config, hyperparameters, global_config
from utils.align_data import pre_process_images
from scripts.run_pti import run_PTI
import matplotlib.pyplot as plt
from scripts.latent_editor_wrapper import LatentEditorWrapper
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics


image_dir_name = 'image'

image_name = 'image'
use_multi_id_training = False
global_config.device = torch.device('cuda')
paths_config.e4e = './PTI/pretrained_models/e4e_ffhq_encode.pt'
paths_config.input_data_id = image_dir_name
paths_config.input_data_path = './PTI/image_processed'
paths_config.checkpoints_dir = './PTI/embeddings'
paths_config.style_clip_pretrained_mappers = './PTI/pretrained_models'
hyperparameters.use_locality_regularization = False


device = torch.device('cuda')

age=0.8

def run(age, model_path):
    paths_config.stylegan2_ada_ffhq = model_path
    ## CONDITIONS - parameter needs changing IF we train with other parameters
    fov_deg = 4.2647 # 18.837 # might be that 4.2647   # FFHQ's FOV
    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    cam_pivot = torch.tensor([0,0,0.2], device=device)
    cam_radius = 2.7
    conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
    camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

    # sæt den der kopierede vektor ind og brug den
    # slet den ene parameter da man kun bruger en (tror jeg)

    # c = torch.cat((conditioning_params, torch.tensor([[age]], device=device)), 1)
    # c_params = torch.cat((camera_params, torch.tensor([[age]], device=device)), 1).float()

    
    c = conditioning_params
    c_params = camera_params

    c = np.array([0.9422833919525146, 0.034289587289094925, 0.3330560326576233, -0.8367999667889383, 0.03984849900007248, -0.9991570711135864, -0.009871904738247395, 0.017018394869192363, 0.33243677020072937, 0.022573914378881454, -0.9428553581237793, 2.566997504832856, 0.0, 0.0, 0.0, 1.0, 4.2647, 0.0, 0.5, 0.0, 4.2647, 0.5, 0.0, 0.0, 1.0])#, -0.4627])
    c = np.reshape(c, (1, 25))
    c = torch.FloatTensor(c).cuda()

    print("Running PTI optimization...")
    model_id = run_PTI(c, use_wandb=False, use_multi_id_training=False)
    print("Finished running PTI optimization")

if __name__ == "__main__":
    run(age)
