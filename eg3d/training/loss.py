# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Loss functions."""

from pydoc import doc
import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
from training.dual_discriminator import filtered_resizing
from training.estimate_age import AgeEstimator, AgeEstimatorNew
import time
from training.face_id import FaceIDLoss
import random
from training.training_loop import normalize
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from numpy.random import uniform
from itertools import combinations
#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, 
                    pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, 
                    blur_init_sigma=0, blur_fade_kimg=0, r1_gamma_init=0, r1_gamma_fade_kimg=0, 
                    neural_rendering_resolution_initial=64, neural_rendering_resolution_final=None, 
                    neural_rendering_resolution_fade_kimg=0, gpc_reg_fade_kimg=1000, gpc_reg_prob=None, 
                    dual_discrimination=False, filter_mode='antialiased', age_version='v2', 
                    age_min=0, age_max=100, id_model="FaceNet", alternate_losses=False, alternate_after=100000,
                    initial_age_training=0, age_loss_fn = "MSE", crop_before_estimate_ages=False, description=""):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg
        self.r1_gamma_init      = r1_gamma_init
        self.r1_gamma_fade_kimg = r1_gamma_fade_kimg
        self.neural_rendering_resolution_initial = neural_rendering_resolution_initial
        self.neural_rendering_resolution_final = neural_rendering_resolution_final
        self.neural_rendering_resolution_fade_kimg = neural_rendering_resolution_fade_kimg
        self.gpc_reg_fade_kimg = gpc_reg_fade_kimg
        self.gpc_reg_prob = gpc_reg_prob
        self.dual_discrimination = dual_discrimination
        self.filter_mode = filter_mode
        self.resample_filter = upfirdn2d.setup_filter([1,3,3,1], device=device)
        self.blur_raw_target = True
        self.age_loss_MSE = torch.nn.MSELoss()
        self.age_loss_L1 = torch.nn.L1Loss()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.cosine_sim = torch.nn.CosineSimilarity()
        self.id_model = FaceIDLoss(device, model = id_model)
        self.age_loss_fn = age_loss_fn
        self.age_version = age_version
        self.age_min = age_min
        self.age_max = age_max
        if age_version == 'v1':
            self.age_model = AgeEstimator(age_min=self.age_min, age_max=self.age_max)
        elif age_version == "v2":
            self.age_model = AgeEstimatorNew(self.device, age_min=self.age_min, age_max=self.age_max, crop=crop_before_estimate_ages)
        self.alternate_losses = alternate_losses
        self.alternate_after = alternate_after
        self.initial_age_training = initial_age_training
        self.frontal_camera_params = self.get_frontal_camera_params()


        assert self.gpc_reg_prob is None or (0 <= self.gpc_reg_prob <= 1)

    def get_frontal_camera_params(self):
        fov_deg = 18.837
        intrinsics = FOV_to_intrinsics(fov_deg, device=self.device)
        cam_pivot = torch.tensor(self.G.rendering_kwargs.get('avg_camera_pivot', [0,0,0]), device=self.device)
        cam_radius = self.G.rendering_kwargs.get('avg_camera_radius', 2.7)
        angle_y, angle_p = 0,0 # straight angle
        cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=self.device)
        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        # random age is appended to camera params to match dimension
        # the age does not affect the output in the synthesis step, see triplane.py line 51
        c_params = torch.cat((camera_params, torch.tensor([[0]], device=self.device)), 1)
        return c_params

    def run_age_loss(self, imgs, c, loss_fn="MSE"):
        """Returns the age loss given a series of generated images and the age the synthetic images
        are suppose to resemble.

        Args:
            imgs (tensor): tensor of images 
            c (tensor): conditions
            loss (str, optional): loss function. Defaults to "MSE".

        Raises:
            NotImplementedError: If the loss function isn't implemented

        Returns:
            tensor: loss
        """
        images = imgs['image']
        predicted_ages, logits = self.age_model.estimate_age(images)
        predicted_ages = predicted_ages.to(self.device)
        if loss_fn == "MSE":
            ages = c[:,-1].clone()
            loss = self.age_loss_MSE(predicted_ages, ages) 
        elif loss_fn =="MAE" or loss_fn=="L1":
            ages = c[:,-1].clone()
            loss = self.age_loss_L1(predicted_ages, ages)
        elif loss_fn =="CAT":
            ages = c[:,25:].clone()
            loss = self.cross_entropy_loss(logits, ages)

        else:
            raise NotImplementedError
        
        return loss

    def pairwise(self, iterable):
        "s -> (s0, s1), (s2, s3), (s4, s5), ..."
        a = iter(iterable)
        return zip(a, a)

    def run_id_loss2(self, imgs, gen_z, gen_c, loss='MSE'):
        images = imgs['image']
        total_loss = 0
        for img1, img2 in self.pairwise(images): # every pair should have the same latent code and same c except for the age parameter
            latent_coords_1 = self.id_model.get_feature_vector(img1[None, :, :, :]) # to get the proper shape of [1, C, W, H] and not [C, W, H]
            latent_coords_2 = self.id_model.get_feature_vector(img2[None, :, :, :])
            constant = 1e-8
            cos = 1 + self.cosine_sim(latent_coords_1, latent_coords_2) + constant
            loss = - torch.log10(cos) + torch.log10(torch.tensor([2 + constant], device=self.device))
            total_loss += loss

        return total_loss

    def compute_weight(self, delta_age):
        return 0.25 * torch.cos((torch.pi/2) * delta_age) + 0.75

    def run_id_loss3(self, z, c_swapped, neural_rendering_resolution):
        """Third iteration of id loss.
        Takes a random latent code from the batch (limited to one due to memory issues).
        Generates four random ages by sampling uniformly from these intervals:
        ```
         random_ages = [
            uniform(-1, -0.5),
            uniform(-0.5,0),
            uniform(0, 0.5),
            uniform(0.5, 1)
        ]
        ```
        Generates four new images with the same latent code but with four different ages
        
        given by `random_ages`. Extracts feature vectors for each image and then finds id loss
        for every combination of faces.

        The loss is weighted using `compute_weight` so that a large age change has less effect than a smaller one.
        The weight for no age change is 1 and is 0.5 for the maximum of age change of 2 (equivalent of age_min to age_max years).

        Args:
            z (tensor): size [batch, 512]
            c_swapped (tensor): conditions used to generate images in ``run_G``
            neural_rendering_resolution (int): 

        Returns:
            tensor: mean id loss
        """
        new_c_swapped = c_swapped.clone() # used for the G.mapping step so the "scene identity" is preserved
        idx = np.random.randint(0, len(z))
        zi = z[idx]
        if self.age_loss_fn == "MSE": # not using age categories
            id_loss = torch.tensor([], device=self.device)
            
            random_ages = [
                uniform(-1, -0.5),
                uniform(-0.5,0),
                uniform(0, 0.5),
                uniform(0.5, 1)
            ]
            age_ranges = len(random_ages)
            new_c = torch.repeat_interleave(new_c_swapped[idx][None,:], age_ranges, axis=0)
            new_c[:,-1] = torch.tensor(random_ages)
            z_identical = torch.repeat_interleave(zi[None,:], age_ranges, axis=0)
            
            ws = self.G.mapping(z_identical, new_c)
            c_params = torch.repeat_interleave(self.frontal_camera_params, age_ranges, axis=0)
            gen_img = self.G.synthesis(ws, c_params, neural_rendering_resolution=neural_rendering_resolution, update_emas=False)
            new_images = gen_img['image']
            f = self.id_model.get_feature_vector(new_images)
            
            for k, l in list(combinations(list(range(age_ranges)), r=2)):
                loss = 1 - self.cosine_sim(f[k][None,:], f[l][None,:])
                Delta_age=new_c[l, - 1] - new_c[k,-1]
                weight = self.compute_weight(Delta_age)
                weighted_loss = loss * weight
                id_loss = torch.cat((id_loss, weighted_loss),0)
            return id_loss.mean()
        else: #categories
            id_loss = torch.tensor([], device=self.device)
            l = np.linspace(0,self.age_max-(self.age_max // 4),4)
            random_ages_idx = [
                np.random.randint(l[0],l[1]),
                np.random.randint(l[1],l[2]),
                np.random.randint(l[2],l[3]),
                np.random.randint(l[3],self.age_max+1)
            ]
            random_ages = np.array([[0]*101]*4)
            random_ages[[0,1,2,3], random_ages_idx] = 1
            age_ranges = len(random_ages)
            new_c = torch.repeat_interleave(new_c_swapped[idx][None,:], age_ranges, axis=0)
            new_c[:, 25:] = torch.tensor(random_ages)
            z_identical = torch.repeat_interleave(zi[None,:], age_ranges, axis=0)

            ws = self.G.mapping(z_identical, new_c)
            c_params = torch.repeat_interleave(self.frontal_camera_params, age_ranges, axis=0)
            gen_img = self.G.synthesis(ws, c_params, neural_rendering_resolution=neural_rendering_resolution, update_emas=False)
            new_images = gen_img['image']
            f = self.id_model.get_feature_vector(new_images)

            for k, l in list(combinations(list(range(age_ranges)), r=2)):
                loss = 1 - self.cosine_sim(f[k][None,:], f[l][None,:])
                Delta_age = torch.tensor(random_ages_idx[l] - random_ages_idx[k], device=self.device)
                weight = self.compute_weight(Delta_age)
                weighted_loss = loss * weight
                id_loss = torch.cat((id_loss, weighted_loss),0)
            return id_loss.mean()



    def run_id_loss(self, imgs, z, c, c_swapped, neural_rendering_resolution, margin=0.2, loss='MSE', update_emas=False, slope=10):
        """Returns the identity loss of a subject by comparing the given images to the 
        images aged and young-ified. 
        
        """
        images = imgs['image']
        
        if self.age_loss_fn == "MSE": # not using age categories
            ages = c[:,-1].clone()
            random_ages = []

            for age in ages:
                new_age = random.uniform(-1,1)
                while np.abs(age.item() - new_age) < margin:
                    new_age = random.uniform(-1, 1)
                random_ages.append(new_age)

            new_c_swapped = c_swapped.clone() # used for the G.mapping step so the "scene identity" is preserved
            new_c_swapped[:,-1] = torch.tensor(random_ages)

            new_c = c.clone() # used for G.synthesis so that the camera angle is preserved
            new_c[:, -1] = torch.tensor(random_ages)

        else: # ages are categorized
            ages = c[:,25:].clone() # categories size [batch_size, len(categories) - 1]
            categories = list(range(101)) 
            left = torch.bucketize(self.age_min, torch.tensor(categories, device = self.device), right=False).item()
            right = torch.bucketize(self.age_max, torch.tensor(categories, device = self.device), right=True).item()

            j = torch.arange(ages.size(0)).long().to(self.device) # used to index

            number_of_age_ranges = right - left

            p = 1/(number_of_age_ranges - 1) # probability of choosing another category
            probabilities = torch.full_like(ages, p).to(self.device)
            probabilities[ages.bool()] = 0 # uniform probability of choosing another category except for the one it was before
            probabilities[j,:left]=0 # dont pick ranges less than age_min
            probabilities[j,right:]=0 # dont pick ranges bigger than age_max
            idx = probabilities.multinomial(1, False).flatten().to(self.device) # choosing the index of the new category
            new_categories = torch.zeros_like(ages).to(self.device)
            new_categories[j, idx] = 1

            new_c_swapped = c_swapped.clone() # used for the G.mapping step so the "scene identity" is preserved
            new_c_swapped[:, 25:] = new_categories

            new_c = c.clone() # used for G.synthesis so that the camera angle is preserved
            new_c[:, 25:] = new_categories

        ws = self.G.mapping(z, new_c_swapped)

        gen_img = self.G.synthesis(ws, new_c, neural_rendering_resolution=neural_rendering_resolution, update_emas=update_emas)
        new_images = gen_img['image']

        latent_coords = self.id_model.get_feature_vector(images)
        new_latent_coords = self.id_model.get_feature_vector(new_images)

        if loss=='MSE':
            return self.age_loss_MSE(latent_coords, new_latent_coords)
        elif loss == 'cosine_similarity_mod':
            cos = self.cosine_sim(latent_coords, new_latent_coords)
            l = - torch.tensor([slope], device=self.device) * cos + torch.tensor([slope], device=self.device) # linear loss
            return l.mean()
        elif loss =='cosine_similarity':
            cos = self.cosine_sim(latent_coords, new_latent_coords)
            return -cos.mean() + 1
        else:
            raise NotImplementedError

    def alternate_scales(self, cur_nimg, age_scale, id_scale):
        """Help alternate between training aging and ID preservation.

        Args:
            cur_nimg (int): current number of images trained on
            age_scale (float): predefined age_scale
            id_scale (float): predefined id_scale
            inital_age_training_nimg (int, optional): How many initial images should be used to only train aging. Defaults to 200000.
            alternate_loop (int, optional): Total number of images used to train both aging and ID preservation
            in an "alternating loop" The first alternate_loop/2 are used to train ID the next alternate_loop/2 to train aging. Defaults to 200000.

        Returns:
            (age_scale, id_scale): updated scaling parameters
        """
        
        alternate_loop = 2 * self.alternate_after
        if id_scale != 0:
            if self.alternate_losses:
                if cur_nimg > self.initial_age_training:
                    if (cur_nimg % alternate_loop) < alternate_loop//2:
                        # train id
                        age_scale = 0
                    else:
                        # train age
                        id_scale = 0
                else:
                    id_scale = 0
            else:
                if cur_nimg < self.initial_age_training:
                    id_scale = 0
        return age_scale, id_scale

    def run_G(self, z, c, swapping_prob, neural_rendering_resolution, update_emas=False):
        if swapping_prob is not None:
            c_swapped = torch.roll(c.clone(), 1, 0)
            c_gen_conditioning = torch.where(torch.rand((c.shape[0], 1), device=c.device) < swapping_prob, c_swapped, c)
            # will swap the items in the c vector if the torch.rand is greater than swapping_prob
        else:
            c_gen_conditioning = torch.zeros_like(c)

        ws = self.G.mapping(z, c_gen_conditioning, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        gen_output = self.G.synthesis(ws, c, neural_rendering_resolution=neural_rendering_resolution, update_emas=update_emas)
        return gen_output, ws, c_gen_conditioning

    def run_D(self, img, c, blur_sigma=0, blur_sigma_raw=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img['image'].device).div(blur_sigma).square().neg().exp2()
                img['image'] = upfirdn2d.filter2d(img['image'], f / f.sum())

        if self.augment_pipe is not None:
            augmented_pair = self.augment_pipe(torch.cat([img['image'],
                                                    torch.nn.functional.interpolate(img['image_raw'], size=img['image'].shape[2:], mode='bilinear', antialias=True)],
                                                    dim=1))
            img['image'] = augmented_pair[:, :img['image'].shape[1]]
            img['image_raw'] = torch.nn.functional.interpolate(augmented_pair[:, img['image'].shape[1]:], size=img['image_raw'].shape[2:], mode='bilinear', antialias=True)

        logits = self.D(img, c, update_emas=update_emas)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg, age_scale=1, age_loss_fn="MSE", id_scale = 1):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if self.G.rendering_kwargs.get('density_reg', 0) == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        r1_gamma = self.r1_gamma

        alpha = min(cur_nimg / (self.gpc_reg_fade_kimg * 1e3), 1) if self.gpc_reg_fade_kimg > 0 else 1
        swapping_prob = (1 - alpha) * 1 + alpha * self.gpc_reg_prob if self.gpc_reg_prob is not None else None

        if self.neural_rendering_resolution_final is not None:
            alpha = min(cur_nimg / (self.neural_rendering_resolution_fade_kimg * 1e3), 1)
            neural_rendering_resolution = int(np.rint(self.neural_rendering_resolution_initial * (1 - alpha) + self.neural_rendering_resolution_final * alpha))
        else:
            neural_rendering_resolution = self.neural_rendering_resolution_initial

        real_img_raw = filtered_resizing(real_img, size=neural_rendering_resolution, f=self.resample_filter, filter_mode=self.filter_mode)

        if self.blur_raw_target:
            blur_size = np.floor(blur_sigma * 3)
            if blur_size > 0:
                f = torch.arange(-blur_size, blur_size + 1, device=real_img_raw.device).div(blur_sigma).square().neg().exp2()
                real_img_raw = upfirdn2d.filter2d(real_img_raw, f / f.sum())

        real_img = {'image': real_img, 'image_raw': real_img_raw}

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                #initialize losses
                age_loss, id_loss = torch.Tensor([0]).to(self.device), torch.Tensor([0]).to(self.device)

                age_scale, id_scale = self.alternate_scales(cur_nimg, age_scale, id_scale)

                gen_img, gen_ws, c_swapped = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                if age_scale != 0:
                    age_loss = self.run_age_loss(gen_img, c_swapped, loss_fn=age_loss_fn)
                age_loss_scaled = age_loss * age_scale # age scaling
                
                if id_scale != 0:
                    id_loss = self.run_id_loss3(gen_z, c_swapped, neural_rendering_resolution)

                id_loss_scaled = id_loss * id_scale # id scaling
                loss_Gmain = torch.nn.functional.softplus(-gen_logits)

                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                training_stats.report('Loss/scores/age', age_loss_scaled)
                training_stats.report('Loss/scores/id', id_loss_scaled)
                training_stats.report('Loss/G/loss', loss_Gmain)
                
            with torch.autograd.profiler.record_function('Gmain_backward'):
                (loss_Gmain.mean() + (age_loss_scaled) + (id_loss_scaled)).mul(gain).backward() # added age loss and id loss
                
        # Density Regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'l1':
            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
            initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * self.G.rendering_kwargs['density_reg_p_dist']
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
            TVloss.mul(gain).backward()

        # Alternative density regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'monotonic-detach':
            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)

            initial_coordinates = torch.rand((ws.shape[0], 2000, 3), device=ws.device) * 2 - 1 # Front

            perturbed_coordinates = initial_coordinates + torch.tensor([0, 0, -1], device=ws.device) * (1/256) * self.G.rendering_kwargs['box_warp'] # Behind
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            monotonic_loss = torch.relu(sigma_initial.detach() - sigma_perturbed).mean() * 10
            monotonic_loss.mul(gain).backward()


            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
            initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * (1/256) * self.G.rendering_kwargs['box_warp']
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
            TVloss.mul(gain).backward()

        # Alternative density regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'monotonic-fixed':
            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)

            initial_coordinates = torch.rand((ws.shape[0], 2000, 3), device=ws.device) * 2 - 1 # Front

            perturbed_coordinates = initial_coordinates + torch.tensor([0, 0, -1], device=ws.device) * (1/256) * self.G.rendering_kwargs['box_warp'] # Behind
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            monotonic_loss = torch.relu(sigma_initial - sigma_perturbed).mean() * 10
            monotonic_loss.mul(gain).backward()


            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
            initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * (1/256) * self.G.rendering_kwargs['box_warp']
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
            TVloss.mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws, c_swapped = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, update_emas=True)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits)
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp_image = real_img['image'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp_image_raw = real_img['image_raw'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp = {'image': real_img_tmp_image, 'image_raw': real_img_tmp_image_raw}

                real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    if self.dual_discrimination:
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image'], real_img_tmp['image_raw']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                            r1_grads_image_raw = r1_grads[1]
                        r1_penalty = r1_grads_image.square().sum([1,2,3]) + r1_grads_image_raw.square().sum([1,2,3])
                    else: # single discrimination
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                        r1_penalty = r1_grads_image.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
