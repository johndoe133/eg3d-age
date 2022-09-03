import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import PIL.Image
from PIL import ImageDraw, ImageFont
import torch
from tqdm import tqdm
import mrcfile

from networks.DEX.estimate_age import AgeEstimator2
import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from training.triplane import TriPlaneGenerator
from training.training_loop import denormalize
import imageio

@click.command()
@click.option('--network_folder', help='Network folder name', required=True)
@click.option('--network', help='Network folder name', default=None, required=False)
@click.option('--seed', type=int, help='Random seed (e.g. 42)', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--shapes', help='Export shapes as .mrc files viewable in ChimeraX', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--shape-res', help='', type=int, required=False, metavar='int', default=512, show_default=True)
@click.option('--fov-deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=18.837, show_default=True)
@click.option('--shape-format', help='Shape Format', type=click.Choice(['.mrc', '.ply']), default='.mrc')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--img_h', help='Height/width of image', default=512, required=False)
def generate_images(
    network_folder: str,
    network: str,
    seed: int,
    truncation_psi: float,
    truncation_cutoff: int,
    shapes: bool,
    shape_res: int,
    fov_deg: float,
    shape_format: str,
    class_idx: Optional[int],
    reload_modules: bool,
    img_h: int,
):
    print(f'Loading networks from "{network_folder}"...')
    device = torch.device('cuda')
    if network is not None:
        network_pkl = network
    else:
        pkls = [string for string in os.listdir(network_folder) if '.pkl' in string]
        pkls = sorted(pkls)
        network_pkl = pkls[-1]
    network_pkl = os.path.join(network_folder, network_pkl)

    outdir = network_folder

    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

    def normalize(x, rmin = 5, rmax = 80, tmin = -1, tmax = 1):
        # normalize age between -1 and 1
        z = ((x - rmin) / (rmax - rmin)) * (tmax - tmin) + tmin
        return z

    fov_deg = 18.837
    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)

    imgs = []

    angle_p = -0.2
    ages = [5,10,20,30,40,50,60,70]
    ages = [normalize(age) for age in ages]

    for angle_y, angle_p in [(.4, angle_p), (0, angle_p), (-.4, angle_p)]:
        cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
        cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
        cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
        conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

        
        for age in ages:
            cuda0 = torch.device('cuda:0')
            c = torch.cat((conditioning_params, torch.tensor([[age]], device=cuda0)), 1)
            c_params = torch.cat((camera_params, torch.tensor([[age]], device=cuda0)), 1)
            ws = G.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
            img = G.synthesis(ws, c_params)['image']

            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            imgs.append(img)
    angles = 3
    no_ages = len(ages)
    img = torch.cat(imgs, dim=2)
    #t = torch.zeros((128*angles, 128*no_ages, 3))
    img_stack=[]
    for i in range(angles):
        img_stack.append(img[0][:, i*img_h*no_ages: (i+1)*img_h*no_ages,:])
    t = torch.cat(img_stack)
    PIL.Image.fromarray(t.cpu().numpy(), 'RGB').save(f'{network_folder}/seed{seed:04d}.png')
    print(f'saved at {outdir}/seed{seed:04d}.png')

    ######## GIF #########
    print("Creating .gif file...")
    ages = np.linspace(-1,1,75) #one for each age
    imgs_gif = []
    font = ImageFont.truetype("FreeSerif.ttf", 40)
    
    angle_y, angle_p = (0, -0.2)
    cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
    cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
    conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
    camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

    for age in ages:
        cuda0 = torch.device('cuda:0')
        c = torch.cat((conditioning_params, torch.tensor([[age]], device=cuda0)), 1)
        c_params = torch.cat((camera_params, torch.tensor([[age]], device=cuda0)), 1)
        c_params = c_params.float()
        ws = G.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        img = G.synthesis(ws, c_params)['image']

        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        pil_img = PIL.Image.fromarray(img[0,:,:,:].cpu().numpy().astype('uint8')) # to draw on
        text_added = ImageDraw.Draw(pil_img)
        text_color="#A0240A"
        text_added.text((0,450), f"Age: {int(denormalize(age))}", font=font, fill=text_color)
        imgs_gif.append(np.array(pil_img))

    # imgs_gif = [tensor.cpu().numpy()[0,:,:,:] for tensor in imgs_gif]
    print("Saving gif..")
    imageio.mimsave(f'{network_folder}/seed{seed:04d}.gif', imgs_gif)
    print("Exiting..")

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter