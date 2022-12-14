# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import copy
import wandb
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from configs import global_config, hyperparameters
from utils import log_utils
import dnnlib
# from camera_utils import LookAtPoseSampler
from PIL import Image
import os
# from camera_utils import LookAtPoseSampler, FOV_to_intrinsics


def project(
        G,
        target: torch.Tensor,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
        c,
        image_name,
        trunc,
        *,
        num_steps=1000,
        z_avg_samples=10000,
        initial_learning_rate=0.01,
        initial_noise_factor=0.05,
        lr_rampdown_length=0.25,
        lr_rampup_length=0.05,
        noise_ramp_length=0.75,
        regularize_noise_weight=1e5,
        verbose=False,
        device: torch.device,
        use_wandb=False,
        initial_z=None,
        image_log_step=global_config.image_rec_result_log_snapshot,
        w_name: str
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device).float()  # type: ignore

    # Compute w stats.
    logprint(f'Computing Z midpoint and stddev using {z_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(z_avg_samples, G.z_dim) # size (600, 512)
    z_avg = np.mean(z_samples, axis = 0)
    z_std = (np.sum((z_samples - z_avg) ** 2) / z_avg_samples) ** 0.5 # scalar
    c_repeat = c.repeat(z_avg_samples, 1) # 600, 26


    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), c_repeat)  # [N, L, C] torch.Size([600, 14, 512])
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C] (600, 1, 512)
    w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C] shape (1, 1, 512)
    w_avg_tensor = torch.from_numpy(w_avg).to(global_config.device) 
    w_std = (np.sum((w_samples - w_avg) ** 2) / z_avg_samples) ** 0.5 # scalar

    start_z = initial_z if initial_z is not None else z_avg
    # start_w size=(1,1,512)
    # Setup noise inputs.
    noise_bufs = {name: buf for (name, buf) in G.backbone.synthesis.named_buffers() if 'noise_const' in name}

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    # Target image is the one we try to match with the optimization, target.shape: torch.Size([3, 512, 512])
    target_images = target.unsqueeze(0).to(device).to(torch.float32)  # torch.Size([1, 3, 512, 512])
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area') # torch.Size([1, 3, 256, 256])
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)
    start_z = start_z[None, :] # [1,512]
    z_opt = torch.tensor(start_z, dtype=torch.float32, device=device, requires_grad=True)  # torch.Size([512])
                            # pylint: disable=not-callable
    age = c[:,-1].cpu().numpy()
    age_opt = torch.tensor(age, dtype=torch.float32, device=device, requires_grad=True)
    P = c[:,:-1]
    
    optimizer = torch.optim.Adam([z_opt] + list(noise_bufs.values()) + [age_opt], betas=(0.9, 0.999),
                                    lr=hyperparameters.first_inv_lr)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True
    images = []
    for step in tqdm(range(num_steps)):

        # Learning rate schedule.
        t = step / num_steps
        z_noise_scale = z_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        z_noise = torch.randn_like(z_opt) * z_noise_scale
        zs = (z_opt + z_noise) #.repeat([G.backbone.mapping.num_ws, 1])
        c = torch.cat([P[0], age_opt])[None,:]
        
        ws = G.mapping(zs, c, truncation_psi=trunc)
        synth_images = G.synthesis(ws, c, noise_mode='const', force_fp32=True)['image'] 

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255 / 2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        if step % (num_steps//6) == 0 or step==num_steps: # append to save images
            save_img = (synth_images.permute(0, 2, 3, 1)).clamp(0, 255).to(torch.uint8)
            images.append(save_img)

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight

        if step % image_log_step == 0:
            with torch.no_grad():
                if use_wandb:
                    global_config.training_step += 1
                    wandb.log({f'first projection _{w_name}': loss.detach().cpu()}, step=global_config.training_step)
                    # log_utils.log_image_from_w(w_opt.repeat([1, G.backbone.mapping.num_ws, 1]), G, w_name)

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step + 1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    # save progress image
    home_dir = os.path.expanduser('~')
    path = f"Documents/eg3d-age/eg3d/PTI/output/{image_name}"
    save_name = os.path.join(home_dir, path)
    os.makedirs(save_name, exist_ok=True)
    img = torch.cat(images, dim=2)
    Image.fromarray(img[0].cpu().numpy(), 'RGB').save(save_name + "/initial_optimization.png")


    G_map_num_ws = G.backbone.mapping.num_ws
    del G
    return z_opt, age_opt
