from math import trunc
import click 
import json
import torch
import dnnlib
import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from training.estimate_age import AgeEstimator, AgeEstimatorNew
from training.training_loop import normalize, denormalize
from tqdm import tqdm
from training.face_id import FaceIDLoss
from scipy.stats import gaussian_kde
from training.coral import Coral
from plot_training_results import plot_setup, compute_figsize
from train import PythonLiteralOption
from torchvision.utils import make_grid
from training.training_loop import get_age_category

def image_grid(imgs, rows, cols):
    #https://stackoverflow.com/questions/37921295/python-pil-image-make-3x3-grid-from-sequence-images
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def generate_data(
    save_name: str,
    network_folder: str,
    network_pkl_path: str,
    network_pkl: str,
    seed: int,
    truncation_psi: float,
    truncation_cutoff: int,
    angles_plot_iterations: int,
    age_model_name: str,
    angles_p: list,
    angles_y: list,
    scatter_iterations: int,
    id_plot_iterations: int,

    ):
    ## LOADING NETWORK ##
    print(f'Loading networks from "{network_folder}"...')
    device = torch.device('cuda')
    
    
    print("Loading network named:", network_pkl)

    with dnnlib.util.open_url(network_pkl_path) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    np.random.seed(seed) # setting the seed to get consistent results

    print("Generating age data...")
    ages = generate_angles_data(G, age_model_name, device, angles_p, angles_y, angles_plot_iterations, save_name, truncation_cutoff, truncation_psi, network_folder)
    print("Generating id data...")
    ages_id = generate_id_data(G, device, id_plot_iterations, save_name, truncation_cutoff, truncation_psi, network_folder)
    print("Generating scatter plot data...")
    generate_scatter_data(G, device, seed, save_name, network_folder, truncation_cutoff, truncation_psi, age_model_name, scatter_iterations)
    print("Generating evaluation image...")
    generate_image(G, seed, device, network_folder, save_name)
    del G
    return ages, ages_id

def generate_scatter_data(G, device, seed, save_name, network_folder, truncation_cutoff, truncation_psi, age_model_name, iterations):
    ## Age evaluation
    if age_model_name == 'coral':
        age_model = Coral()
    elif age_model_name == 'v1':
        age_model = AgeEstimator()
    elif age_model_name == 'v2':
        age_model = AgeEstimatorNew(device)

    magface = FaceIDLoss(device, model="MagFace")

    training_option_path = os.path.join(network_folder, "training_options.json")
    f = open(training_option_path)
    training_option = json.load(f)
    age_loss_fn = training_option['age_loss_fn']
    age_min = training_option['age_min']
    age_max = training_option['age_max']

    angles_p = np.random.RandomState(seed).uniform(-0.5, 0.5, size=(iterations))
    angles_y = np.random.RandomState(seed+1).uniform(-0.5, 0.5, size=(iterations))
    z = torch.from_numpy(np.random.RandomState(seed).randn(iterations, G.z_dim)).to(device)

    if age_loss_fn == "CAT":
        categories = 101
        ages = np.zeros((iterations, categories))
        for i, _ in enumerate(ages):
            age = np.random.randint(age_min, age_max+1)
            ages[i][age] = 1
    elif age_loss_fn == "MSE":
        ages = np.random.RandomState(seed+2).uniform(-1, 1, size=(iterations, 1))

    fov_deg = 18.837
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0,0,0]), device=device)
    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
    res = []
    for zi, age, angle_p, angle_y in tqdm(zip(z, ages, angles_p, angles_y), total=iterations):
        cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
        conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)    
        c = torch.cat((conditioning_params, torch.tensor([age], device=device)), 1)
        c_params = torch.cat((camera_params, torch.tensor([age], device=device)), 1).float()

        ws = G.mapping(zi[None,:], c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        img = G.synthesis(ws, c_params)['image']
        age_hat, logits = age_model.estimate_age(img)

        features = magface.get_feature_vector(img)
        mag = np.linalg.norm(features.cpu().numpy())
        res.append([age_hat.item(), age[0], angle_p, angle_y, mag])

    columns = ["age_hat", "age_true", "angle_p", "angle_y", "mag"]
    df = pd.DataFrame(res, columns=columns)
    # Save as csv file
    save_dir = os.path.join("Evaluation","Runs", save_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "age_scatter.csv")
    print("Saving csv at", save_dir,"...")
    df.to_csv(save_path, index=False)   

def generate_id_data(
        G, device, angles_plot_iterations, save_name, truncation_cutoff, truncation_psi, network_folder
    ):
    seeds = np.random.randint(1,100000, size = angles_plot_iterations)

    training_option_path = os.path.join(network_folder, "training_options.json")
    f = open(training_option_path)
    training_option = json.load(f)
    training_id_model = training_option['loss_kwargs']['id_model']
    id_model = "MagFace" if training_id_model=="ArcFace" else "ArcFace"
    cosine_sim_f = torch.nn.CosineSimilarity()

    age_min = training_option['age_min']
    age_max = training_option['age_max']

    age_loss_fn = training_option['age_loss_fn']
    ages = list(map(int, np.linspace(age_min, age_max, 5)))
    id_model = FaceIDLoss(device)
    res = []
    for seed in tqdm(seeds):
        for age1 in ages:
            angle_y = np.random.uniform(-0.5,0.5)
            angle_p = np.random.uniform(-0.5,0.5)
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
            c = get_conditioning_parameter(age1, G, device, age_loss_fn)
            c_camera = get_camera_parameters(age1, G, device, angle_y, angle_p, age_loss_fn)
            ws = G.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
            generated_image_1 =  G.synthesis(ws, c_camera)['image']
            feature_v_1 = id_model.get_feature_vector(generated_image_1)
            for age2 in ages:
                if age1 == age2:
                    continue # skip comparing similar images
                c = get_conditioning_parameter(age2, G, device, age_loss_fn)
                c_camera = get_camera_parameters(age2, G, device, angle_y, angle_p, age_loss_fn)
                ws = G.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
                generated_image_2 =  G.synthesis(ws, c_camera)['image']
                feature_v_2 = id_model.get_feature_vector(generated_image_2)
                cosine_sim = cosine_sim_f(feature_v_1, feature_v_2)
                res.append([seed, age1, age2, angle_y, angle_p, cosine_sim.item()])

    # create dataframe
    columns = ["seed", "age1", "age2", "angle_y", "angle_p", "cosine_sim"]
    df = pd.DataFrame(res, columns=columns)

    # Save as csv file
    save_dir = os.path.join("Evaluation","Runs", save_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "id_evaluation.csv")
    print("Saving csv at", save_dir,"...")
    df.to_csv(save_path, index=False)
    return ages

def generate_angles_data(G, age_model_name, device, angles_p, angles_y, angles_plot_iterations, save_name, truncation_cutoff, truncation_psi, network_folder):
    """Will generate `angles_plot_iterations` z-codes and generate images for each combination of `angles_p` and `angles_y`. 

    Args:
        G: generator
        age_model_name (str): 
        device: 
        angles_p (list): _description_
        angles_y (list): _description_
        angles_plot_iterations (int): _description_
        save_name (_type_): _description_
        truncation_cutoff (_type_): _description_
        truncation_psi (_type_): _description_
        network_folder (_type_): _description_
    """
    seeds = np.random.randint(1,100000, size = angles_plot_iterations)

    training_option_path = os.path.join(network_folder, "training_options.json")
    f = open(training_option_path)
    training_option = json.load(f)
    age_min = training_option['age_min']
    age_max = training_option['age_max']
    age_loss_fn = training_option['age_loss_fn']
    ## Age evaluation
    if age_model_name == 'coral':
        age_model = Coral()
    elif age_model_name == 'v1':
        age_model = AgeEstimator()
    elif age_model_name == 'v2':
        age_model = AgeEstimatorNew(device)

    ages = list(map(int, np.linspace(age_min, age_max, 9))) # 9 to fit a 3x3 grid
    angles = []
    for angle_p in angles_p:
        for angle_y in angles_y:
            angles.append((angle_y, angle_p))
        res = []
    for seed in tqdm(seeds):
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        for age in ages:
            c = get_conditioning_parameter(age, G, device, age_loss_fn)
            for angle_y, angle_p in angles:
                c_camera = get_camera_parameters(age, G, device, angle_y, angle_p, age_loss_fn)
                ws = G.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
                generated_image =  G.synthesis(ws, c_camera)['image']
                age_hat, logits = age_model.estimate_age(generated_image)
                if age_model_name in ["v1", "v2"]:
                    age_hat = denormalize(age_hat)
                age_hat = age_hat.item()
                mae = np.abs(age - age_hat)
                error = age-age_hat
                res.append([seed, age, angle_y, angle_p, age_hat, mae, error])
    
    # create dataframe
    columns = ["seed", "age", "angle_y", "angle_p", "age_hat", "mae", "error"]
    df = pd.DataFrame(res, columns=columns)

    # Save as csv file
    save_dir = os.path.join("Evaluation", "Runs", save_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir,f"age_evaluation.csv")
    print("Saving csv at", save_dir,"...")
    df.to_csv(save_path, index=False)
    return ages

def generate_image(G, seed, device, path, save_name):
    training_option_path = os.path.join(path, "training_options.json")
    f = open(training_option_path)
    training_option = json.load(f)
    age_loss_fn = training_option['age_loss_fn']
    age_min = training_option['age_min']
    age_max = training_option['age_max']

    angles_y = [0.5, 0, -0.5]
    angles_p = [0.5, 0.25, 0, -0.25, -0.5]
    fov_deg = 18.837

    intrinsics = FOV_to_intrinsics(fov_deg, device=device)

    truncation_psi=1
    truncation_cutoff=14
    font = ImageFont.truetype("FreeSerif.ttf", 72)
    text_color="#FFFFFF"
    images = []
    images_per_angle=2
    cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0,0,0]), device=device)
    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
    for angle_p in angles_p:
        for angle_y in angles_y:
            for i in range(images_per_angle): #generate two images per angle
                z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)
                age_g = np.random.randint(age_min, age_max + 1)
                if age_loss_fn == "CAT":
                    age = [0] * 101
                    age[age_g] = 1
                elif age_loss_fn == "MSE":
                    age = [normalize(age_g, rmin=age_min, rmax=age_max)]
                cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
                conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
                camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
                conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)    
                c = torch.cat((conditioning_params, torch.tensor([age], device=device)), 1)
                c_params = torch.cat((camera_params, torch.tensor([age], device=device)), 1).float()

                ws = G.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
                img = G.synthesis(ws, c_params)['image']
                img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)

                pil_img = Image.fromarray(img.permute(0,2,3,1)[0].cpu().numpy(), 'RGB')
                text_added = ImageDraw.Draw(pil_img)
                
                text_added.text((0,420), f"Age: {age_g}", font=font, fill=text_color)
                images.append(pil_img)
                
                # images.append(img[0])

    save_dir = os.path.join("Evaluation","Runs", save_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "image.png")
    print("Saving image at", save_dir,"...")

    grid = image_grid(images, rows=len(angles_p), cols = images_per_angle*len(angles_y))
    grid.save(save_path)


def get_conditioning_parameter(age, G, device, age_loss_fn, fov_deg = 18.837):
    """Get conditioning parameters for a given age, looking at the camera

    Args:
        age (int)
        G : generator

        device
        fov_deg (float, optional): Field of view. Defaults to 18.837.

    Returns:
        tensor: conditioning parameters
    """
    if age_loss_fn == "MSE":
        age_c = [normalize(age)]
    elif age_loss_fn == "CAT":
        l = [0] * 101
        l[age] = 1
        age_c = np.array(l)
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
    conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
    conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    conditioning_params = torch.cat((conditioning_params, torch.tensor([age_c], device=device)), 1) # add age
    return conditioning_params.float()

def get_camera_parameters(age, G, device, angle_y, angle_p, age_loss_fn, fov_deg = 18.837):
    """Get camera params to rotate the camera angle to change how we look at the synthetic person.
    Could also be seen as rotating the synthetic person.
    Age will have little to no effect in the G.synthesis step but should still be passed as an 
    argument. 

    Args:
        age (float): age of synthetic person
        G: generator
        device: cuda
        angle_y (float): yaw of synthetic person
        angle_p (float): pitch of synthetic person
        fov_deg (float, optional): Field of view. Defaults to 18.837.
    Returns:
        tensor: camera parameters
    """
    if age_loss_fn == "MSE":
        age_c = [normalize(age)]
    elif age_loss_fn == "CAT":
        l = [0] * 101
        l[age] = 1
        age_c = np.array(l)

    intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
    cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
    camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    camera_params = torch.cat((camera_params, torch.tensor([age_c], device=device)), 1)
    return camera_params.float()

if __name__ == "__main__":
    generate_data()