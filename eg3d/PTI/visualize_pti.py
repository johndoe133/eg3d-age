import os
import sys
import pickle
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from configs import paths_config, hyperparameters, global_config
from utils.align_data import pre_process_images
from scripts.run_pti import run_PTI
import matplotlib.pyplot as plt
from scripts.latent_editor_wrapper import LatentEditorWrapper
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
import cv2


def load_generators(image_name):
    with open(paths_config.stylegan2_ada_ffhq, 'rb') as f:
        old_G = pickle.load(f)['G_ema'].cuda()

    embedding_dir_G = './PTI/embeddings/G'
    with open(f'{embedding_dir_G}/{image_name}.pt', 'rb') as f_new: 
        new_G = torch.load(f_new).cuda()

    return old_G, new_G

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def visualize(image_name, c, trunc):
    old_G, new_G = load_generators(image_name)

    embedding_dir_w = '/work3/morbj/embeddings/w'
    z_pivot = torch.load(f'{embedding_dir_w}/{image_name}.pt')

    w_pivot_old = old_G.mapping(z_pivot, c, truncation_psi=trunc)
    old_image = old_G.synthesis(w_pivot_old, c)['image']

    w_pivot = new_G.mapping(z_pivot, c, truncation_psi=trunc)
    new_image = new_G.synthesis(w_pivot, c)['image'] #, noise_mode='const', force_fp32 = True

    images = [old_image, new_image]
    names = ["initial_inversion", "pti_inversion"]
    font_size = 55
    for i, img in enumerate(images): 
        invertion = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()[0] 
        pil_invertion = Image.fromarray(invertion)

        original_img_path = os.path.join(paths_config.input_data_path, image_name + '.png')
        original_img = cv2.imread(original_img_path) # load image
        original_img =  cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        original_img = Image.fromarray(original_img)

        pil_img = get_concat_h(original_img, pil_invertion)

        draw = ImageDraw.Draw(pil_img)
        font = ImageFont.truetype("FreeSerif.ttf", font_size)

        draw.text((0, 505-font_size), f"Original", (255,255,255), font=font)
        draw.text((512, 505-font_size), f"Reconstruction", (255,255,255), font=font)
        output_dir = f"./PTI/output/{image_name}"
        os.makedirs(output_dir, exist_ok=True)
        pil_img.save(f"{output_dir}/{names[i]}.png")