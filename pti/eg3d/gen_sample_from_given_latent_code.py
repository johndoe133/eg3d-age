# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate lerp videos using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import scipy.interpolate
import torch
import PIL.Image
import legacy

from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from training.triplane import TriPlaneGenerator


# ----------------------------------------------------------------------------

def generate_image(G, latent, truncation_psi, truncation_cutoff, cfg, image_mode, device, output, fov_deg):
    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)

    ws = latent # 1, 14, 512                                                                                                  device=device)

    imgs = []
    angle_p = -0.2

    for angle_y, angle_p in [(.4, angle_p), (0, angle_p), (-.4, angle_p)]:
        cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
        cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
        cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
        conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

        img = G.synthesis(ws, camera_params)['image']
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        imgs.append(img)
    img = torch.cat(imgs, dim=2)
    print(f"Saving at: {output}")
    PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(output)


# ----------------------------------------------------------------------------



@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--npy_path', 'npy_path', help='Network pickle filename', required=True)
@click.option('--num-keyframes', type=int,
              help='Number of seeds to interpolate through.  If not specified, determine based on the length of the seeds array given by --seeds.',
              default=None)
@click.option('--w-frames', type=int, help='Number of frames to interpolate between latents', default=120)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--outdir', help='Output directory', type=str, required=True, metavar='DIR')
@click.option('--cfg', help='Config', type=click.Choice(['FFHQ', 'Cats']), required=False, metavar='STR',
              default='FFHQ', show_default=True)
@click.option('--image_mode', help='Image mode', type=click.Choice(['image', 'image_depth', 'image_raw']),
              required=False, metavar='STR', default='image', show_default=True)
@click.option('--sample_mult', 'sampling_multiplier', type=float,
              help='Multiplier for depth sampling in volume rendering', default=2, show_default=True)
@click.option('--nrr', type=int, help='Neural rendering resolution override', default=None, show_default=True)
@click.option('--fov_deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=18.837, show_default=True)
def run_generate_image(
        network_pkl: str,
        npy_path:str,
        truncation_psi: float,
        truncation_cutoff: int,
        num_keyframes: Optional[int],
        w_frames: int,
        outdir: str,
        cfg: str,
        image_mode: str,
        sampling_multiplier: float,
        nrr: Optional[int],
        fov_deg: float
):
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')

    if 'pkl' in network_pkl:
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

        G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * sampling_multiplier)
        G.rendering_kwargs['depth_resolution_importance'] = int(
            G.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)
    else:
        init_args = ()
        init_kwargs = {'z_dim': 512, 'w_dim': 512, 'mapping_kwargs': {'num_layers': 2}, 'channel_base': 32768,
                       'channel_max': 512, 'fused_modconv_default': 'inference_only',
                       'rendering_kwargs': {'depth_resolution': 48, 'depth_resolution_importance': 48,
                                            'ray_start': 2.25, 'ray_end': 3.3, 'box_warp': 1, 'avg_camera_radius': 2.7,
                                            'avg_camera_pivot': [0, 0, 0.2], 'image_resolution': 512,
                                            'disparity_space_sampling': False, 'clamp_mode': 'softplus',
                                            'superresolution_module': 'training.superresolution.SuperresolutionHybrid8XDC',
                                            'c_gen_conditioning_zero': False, 'c_scale': 1.0,
                                            'superresolution_noise_mode': 'none', 'density_reg': 0.25,
                                            'density_reg_p_dist': 0.004, 'reg_type': 'l1', 'decoder_lr_mul': 1.0,
                                            'sr_antialias': True}, 'num_fp16_res': 0, 'sr_num_fp16_res': 4,
                       'sr_kwargs': {'channel_base': 32768, 'channel_max': 512,
                                     'fused_modconv_default': 'inference_only'}, 'conv_clamp': None, 'c_dim': 25,
                       'img_resolution': 512, 'img_channels': 3}
        rendering_kwargs = {'depth_resolution': 96, 'depth_resolution_importance': 96, 'ray_start': 2.25,
                            'ray_end': 3.3, 'box_warp': 1, 'avg_camera_radius': 2.7, 'avg_camera_pivot': [0, 0, 0.2],
                            'image_resolution': 512, 'disparity_space_sampling': False, 'clamp_mode': 'softplus',
                            'superresolution_module': 'training.superresolution.SuperresolutionHybrid8XDC',
                            'c_gen_conditioning_zero': False, 'c_scale': 1.0, 'superresolution_noise_mode': 'none',
                            'density_reg': 0.25, 'density_reg_p_dist': 0.004, 'reg_type': 'l1', 'decoder_lr_mul': 1.0,
                            'sr_antialias': True}

        # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
        print("Reloading Modules!")
        G = TriPlaneGenerator(*init_args, **init_kwargs).eval().requires_grad_(False).to(device)

        ckpt = torch.load(network_pkl)
        G.load_state_dict(ckpt['G_ema'], strict=False)
        G.neural_rendering_resolution = 128

        G.rendering_kwargs = rendering_kwargs

        G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * sampling_multiplier)
        G.rendering_kwargs['depth_resolution_importance'] = int(
            G.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)

    if nrr is not None: G.neural_rendering_resolution = nrr

    if truncation_cutoff == 0:
        truncation_psi = 1.0  # truncation cutoff of 0 means no truncation anyways
    if truncation_psi == 1.0:
        truncation_cutoff = 14  # no truncation so doesn't matter where we cutoff

    latent  = np.load(npy_path)
    latent = torch.FloatTensor(latent).cuda()
    name = os.path.basename(npy_path)[:-4]
    output = os.path.join(outdir, f'{name}.png')
    print("Generating image...")
    generate_image(G, latent, truncation_psi, truncation_cutoff, cfg, image_mode, device, output, fov_deg)

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    run_generate_image()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------   
