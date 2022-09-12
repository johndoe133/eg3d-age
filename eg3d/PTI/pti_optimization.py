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

def run(age, model_path, image_name):
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

    c = np.array([0.9999064803123474, -0.006213949993252754, -0.012183905579149723, 0.028693876930960493, -0.0060052573680877686, -0.9998359084129333, 0.017090922221541405, -0.04020780808014847, -0.012288108468055725, -0.017016155645251274, -0.9997797012329102, 2.6995481091464293, 0.0, 0.0, 0.0, 1.0, 4.2647, 0.0, 0.5, 0.0, 4.2647, 0.5, 0.0, 0.0, 1.0]) #, -0.4627]
    # c = np.array([0.9979292154312134, -0.004497078713029623, 0.06416412442922592, -0.1661906628986501, -0.00316850608214736, -0.9997788071632385, -0.020792603492736816, 0.06072389919955754, 0.06424344331026077, 0.020546242594718933, -0.9977227449417114, 2.694196219957134, 0.0, 0.0, 0.0, 1.0, 4.2647, 0.0, 0.5, 0.0, 4.2647, 0.5, 0.0, 0.0, 1.0])
    c = np.reshape(c, (1, 25))
    c = torch.FloatTensor(c).cuda()

    print("Running PTI optimization...")
    model_id = run_PTI(c, image_name, use_wandb=False, use_multi_id_training=False)
    print("Finished running PTI optimization")
    return c

if __name__ == "__main__":
    run(age)
