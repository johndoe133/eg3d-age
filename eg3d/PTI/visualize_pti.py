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



def load_generators(image_name):
    with open(paths_config.stylegan2_ada_ffhq, 'rb') as f:
        old_G = pickle.load(f)['G_ema'].cuda()

    embedding_dir_G = './PTI/embeddings/G'
    with open(f'{embedding_dir_G}/{image_name}.pt', 'rb') as f_new: 
        new_G = torch.load(f_new).cuda()

    return old_G, new_G

def visualize(image_name, c):
    old_G, new_G = load_generators(image_name)

    embedding_dir_w = './PTI/embeddings/w'
    w_pivot = torch.load(f'{embedding_dir_w}/{image_name}.pt')


    # cuda0 = torch.device('cuda:0')

    # intrinsics = FOV_to_intrinsics(18.837, device=cuda0) #default value
    # angle_y, angle_p = (0,-0.2)
    # cam_pivot = torch.tensor(old_G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=cuda0)
    # cam_radius = old_G.rendering_kwargs.get('avg_camera_radius', 2.7)
    # cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=cuda0)
    # conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=cuda0)
    # camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    # random_age=0.8
    # c_params = torch.cat((camera_params, torch.tensor([[random_age]], device=cuda0)), 1)

    # old_image = old_G.synthesis(w_pivot, c_params, noise_mode='const', force_fp32 = True)['image']
    # new_image = new_G.synthesis(w_pivot, c_params, noise_mode='const', force_fp32 = True)['image']

    old_image = old_G.synthesis(w_pivot, c, noise_mode='const', force_fp32 = True)['image']
    new_image = new_G.synthesis(w_pivot, c, noise_mode='const', force_fp32 = True)['image']

    images = [old_image, new_image]
    names = ["initial_inversion", "pti_inversion"]
    for i, img in enumerate(images): 
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()[0] 
        pil_img = Image.fromarray(img)
        output_dir = f"./PTI/output/{image_name}"
        os.makedirs(output_dir, exist_ok=True)
        pil_img.save(f"{output_dir}/{names[i]}.png")