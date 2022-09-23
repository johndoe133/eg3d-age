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
    z_pivot = torch.load(f'{embedding_dir_w}/{image_name}.pt')

    w_pivot_old = old_G.mapping(z_pivot, c)
    old_image = old_G.synthesis(w_pivot_old, c)['image']

    w_pivot = new_G.mapping(z_pivot, c)
    new_image = new_G.synthesis(w_pivot, c)['image'] #, noise_mode='const', force_fp32 = True

    images = [old_image, new_image]
    names = ["initial_inversion", "pti_inversion"]
    for i, img in enumerate(images): 
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()[0] 
        pil_img = Image.fromarray(img)
        output_dir = f"./PTI/output/{image_name}"
        os.makedirs(output_dir, exist_ok=True)
        pil_img.save(f"{output_dir}/{names[i]}.png")