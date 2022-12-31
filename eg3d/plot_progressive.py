import os
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np

from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch
from tqdm import tqdm

from training.estimate_age import AgeEstimator
import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from training.triplane import TriPlaneGenerator
from training.training_loop import denormalize, get_age_category
from train import PythonLiteralOption
import imageio
import json

@click.command()
@click.option('--network_folder', help='Path to the training folder of the network', required=False, default='')
@click.option('--network', help='Path to specific network', default=None, required=False)
@click.option('--seed', type=int, help='Random seed (e.g. 42)', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--fov-deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=18.837, show_default=True)
@click.option('--img_h', help='image height', type=int, required=False,default=512, show_default=True)
@click.option('--calibrated', help="whether to use a calibrated target age. Use calibrate_network.py to compute function.", required=False, default=False)
def generate_images(
    network_folder: str,
    network: str,
    seed: int,
    truncation_psi: float,
    truncation_cutoff: int,
    fov_deg: float,
    img_h: int,
    calibrated: bool
):
    print(f'Loading networks from "{network_folder}"...')
    device = torch.device('cuda')
    if network is not None:
        network_path = network
        pkl_name=network.split('-')[-1][:-4]
    else:
        pkls = [string for string in os.listdir(network_folder) if '.pkl' in string]
        pkls = sorted(pkls)
        network_pkl = pkls[-1]
        network_path = os.path.join(network_folder, network_pkl)
        pkl_name = network_pkl.split('-')[-1][:-4]
    
    print("Loading network from path:", network_path)
    network_pkl = network_path
    cal = lambda age: age
    if calibrated:
        cal_path = os.path.join('/'.join(network_path.split("/")[:-1]), f"calibrate-{truncation_psi}.npy")
        a,b = np.load(cal_path)
        cal = lambda age: (age-b)/a
    outdir = network_folder
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

    def normalize(x, rmin = 0, rmax = 75, tmin = -1, tmax = 1):
        # normalize age between -1 and 1
        z = ((x - rmin) / (rmax - rmin)) * (tmax - tmin) + tmin
        return z

    fov_deg = 18.837
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)

    imgs = []

    training_option_path = os.path.join(network_folder, "training_options.json")
    f = open(training_option_path)
    training_option = json.load(f)
    if 'age_min' in training_option.keys():
        age_min = training_option['age_min']
        age_max = training_option['age_max']
    else:
        age_min = 0
        age_max = 100

    age_loss_fn = training_option['age_loss_fn']

    angle_p = 0
    # ages = [20,30,40,50,60,70,80,90,100]
    ages = np.linspace(age_min, age_max, 9)
    # if categories != []:
    #     ages=categories
    # ages = [normalize(age) for age in ages]

    for angle_y, angle_p in [(.4, angle_p), (0, angle_p), (-.4, angle_p)]:
        cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0,0,0]), device=device)
        cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
        cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
        conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)    

        for i,age in enumerate(ages):
            if age_loss_fn == "CAT":
                age_list = [0] *101
                age_list[int(age)] = 1
            else:
                age_list = [normalize(cal(age), rmin=age_min, rmax=age_max)]
            c = torch.cat((conditioning_params, torch.tensor([age_list], device=device)), 1)
            c_params = torch.cat((camera_params, torch.tensor([age_list], device=device)), 1)
            ws = G.mapping(z, c.float(), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
            img = G.synthesis(ws, c_params.float())['image']

            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            imgs.append(img)
    angles = 3
    no_ages = len(ages)
    img = torch.cat(imgs, dim=2)
    #t = torch.zeros((128*angles, 128*no_ages, 3))
    font = ImageFont.truetype("FreeSerif.ttf", 40)
    img_stack=[]
    for i in range(angles):
        img_stack.append(img[0][:, i*img_h*no_ages: (i+1)*img_h*no_ages,:])
    t = torch.cat(img_stack)
    pil_img = Image.fromarray(t.cpu().numpy(), 'RGB')
    x, y = pil_img.size
    pil_img = ImageOps.pad(pil_img, (x, y+80), color="white")
    draw = ImageDraw.Draw(pil_img)
    for i, age in enumerate(ages):
        draw.text(((i*512)+230,0), f"Age: {int(age)}", (0,0,0), font=font)

    if calibrated:
        pil_img.save(f'{network_folder}/network{pkl_name}_seed{seed:04d}-t-{truncation_psi}-cal.png')
        print(f'Saved at {network_folder}/network{pkl_name}_seed{seed:04d}-t-{truncation_psi}-cal.png')
    else:
        pil_img.save(f'{network_folder}/network{pkl_name}_seed{seed:04d}-t-{truncation_psi}.png')
        print(f'Saved at {network_folder}/network{pkl_name}_seed{seed:04d}-t-{truncation_psi}.png')

    ######## GIF #########
    font = ImageFont.truetype("FreeSerif.ttf", 40)

    print("Creating .gif file...")
    ages = np.linspace(0,75, 76) #one for each age
    imgs_gif = []
    
    angle_y, angle_p = (0, 0)
    cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
    cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
    conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
    camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

    for age in ages:
        if age_loss_fn == "MSE":
            age_list = [normalize(cal(age), rmin=age_min, rmax=age_max)]
        elif age_loss_fn == "CAT":
            age_list = [0] * 101
            age_list[int(age)] = 1
        cuda0 = torch.device('cuda:0')
        c = torch.cat((conditioning_params, torch.tensor([age_list], device=cuda0)), 1)
        c_params = torch.cat((camera_params, torch.tensor([age_list], device=cuda0)), 1)
        c_params = c_params.float()
        ws = G.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        img = G.synthesis(ws, c_params)['image']

        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        pil_img = Image.fromarray(img[0,:,:,:].cpu().numpy().astype('uint8')) # to draw on
        text_added = ImageDraw.Draw(pil_img)
        text_color="#FFFFFF"
        text_added.text((0,450), f"Age: {int(age)}", font=font, fill=text_color) #SKAL BRUGE AGE_MIN AGE_MAX
        imgs_gif.append(np.array(pil_img))

    # imgs_gif = [tensor.cpu().numpy()[0,:,:,:] for tensor in imgs_gif]
    print("Saving gif..")
    imageio.mimsave(f'{network_folder}/network{pkl_name}_seed{seed:04d}-t-{truncation_psi}.gif', imgs_gif)
    print(f'Saved at {network_folder} as network{pkl_name}_seed{seed:04d}-t-{truncation_psi}.gif')
    print("Exiting..")
    
if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter