# from visualize_pti import load_generators
from configs import paths_config, hyperparameters, global_config
# from pti_pipeline import normalize
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import os
from PIL import Image, ImageFont, ImageDraw
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics

def normalize(x, rmin = 5, rmax = 80, tmin = -1, tmax = 1):
    """Defined in eg3d.training.training_loop
    """
    z = ((x - rmin) / (rmax - rmin)) * (tmax - tmin) + tmin
    return np.round(z, 4)


def load_generators(image_name):
    with open(paths_config.stylegan2_ada_ffhq, 'rb') as f:
        old_G = pickle.load(f)['G_ema'].cuda()

    embedding_dir_G = './PTI/embeddings/G'
    with open(f'{embedding_dir_G}/{image_name}.pt', 'rb') as f_new: 
        new_G = torch.load(f_new).cuda()

    return old_G, new_G

def image_grid(imgs, rows, cols):
    #https://stackoverflow.com/questions/37921295/python-pil-image-make-3x3-grid-from-sequence-images
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def get_conditioning_parameter(age, G, device, age_loss_fn, age_min, age_max, fov_deg = 18.837):
    """Get conditioning parameters for a given age, looking at the camera

    Args:
        age (int): not normalized
        G : generator

        device
        fov_deg (float, optional): Field of view. Defaults to 18.837.

    Returns:
        tensor: conditioning parameters
    """
    if age_loss_fn == "MSE":
        age_c = [normalize(age, rmin=age_min, rmax=age_max)]
    elif age_loss_fn == "CAT":
        l = [0] * 101
        l[int(age)] = 1
        age_c = np.array(l)
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
    conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
    conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    conditioning_params = torch.cat((conditioning_params, torch.tensor(np.array([age_c]), device=device)), 1)
    return conditioning_params.float()

def get_camera_parameters(age, G, device, angle_y, angle_p, age_loss_fn, age_min, age_max, fov_deg = 18.837):
    """Get camera params to rotate the camera angle to change how we look at the synthetic person.
    Could also be seen as rotating the synthetic person.
    Age will have little to no effect in the G.synthesis step but should still be passed as an 
    argument. 

    Args:
        age (float): age of synthetic person
        G: generator
        device: cuda
        angle_y (float): yaw of synthetic person's face
        angle_p (float): pitch of synthetic person's face
        fov_deg (float, optional): Field of view. Defaults to 18.837.
    Returns:
        tensor: camera parameters
    """
    if age_loss_fn == "MSE":
        age_c = [normalize(age, rmin=age_min, rmax=age_max)]
    elif age_loss_fn == "CAT":
        l = [0] * 101
        l[int(age)] = 1
        age_c = np.array(l)

    intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
    cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
    camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    camera_params = torch.cat((camera_params, torch.tensor(np.array([age_c]), device=device)), 1)
    return camera_params.float()

def edit_age(image_name, model_path, c, trunc):
    old_G, new_G = load_generators(image_name)
    ages = [age for age in np.linspace(-25,90,10)]
    embedding_dir_w = '/work3/morbj/embeddings/w'
    model_folder = '/'.join(map(str,model_path.split("/")[:-1]))
    a, b = np.load(os.path.join(model_folder, f"calibrate-{trunc}.npy"))
    cal = lambda age: (age-b)/a
    z_pivot = torch.load(f'{embedding_dir_w}/{image_name}.pt')
    images = []
    for age in ages:
        new_c = c
        new_c[0][-1] = normalize(cal(age), rmin=hyperparameters.age_min, rmax=hyperparameters.age_max)
        w_pivot = new_G.mapping(z_pivot, c, truncation_psi=trunc)
        new_image = new_G.synthesis(w_pivot, new_c)['image']
        img = (new_image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        img = Image.fromarray(img)
        # F.interpolate(img.float(), [224,224],  mode='bilinear', align_corners=True).shape
        images.append(img)
    rows = 2
    columns = 5
    grid = image_grid(images, rows, columns)
    home_dir = os.path.expanduser('~')
    path = f"Documents/eg3d-age/eg3d/PTI/output/{image_name}"
    draw = ImageDraw.Draw(grid)
    font_size = 80
    font = ImageFont.truetype("FreeSerif.ttf", font_size)
    counter = 0
    for i in range(rows):
        for j in range(columns):
            draw.text((j*512 , i*512 + 500 - font_size), f"Age: {int(ages[counter])}", (255,255,255), font=font)
            counter += 1

    save_name = os.path.join(home_dir, path, "aging_effect.png")
    grid.save(save_name)

    angles_list = [0.8, 0.4, 0, -0.4, -0.8]
    angles=[]
    images = []
    for angle_p in [0]:
        for angle_y in angles_list:
            angles.append((angle_y, angle_p))

    ages = np.linspace(0,90,25)
    for i, (angle_y, angle_p ) in enumerate(angles):
        c_camera = get_camera_parameters(age, new_G, torch.device("cuda"), angle_y, angle_p, "MSE", 0, 75)
        c[0][-1] = normalize(cal(29), rmin=hyperparameters.age_min, rmax=hyperparameters.age_max)
        w_pivot = new_G.mapping(z_pivot, c, truncation_psi=trunc)
        new_image = new_G.synthesis(w_pivot, c_camera)['image']
        img = (new_image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        img = Image.fromarray(img)
        # F.interpolate(img.float(), [224,224],  mode='bilinear', align_corners=True).shape
        images.append(img)

    rows = 1
    columns = 5
    grid = image_grid(images, rows, columns)
    save_name = os.path.join(home_dir, path, "angles.png")
    grid.save(save_name)