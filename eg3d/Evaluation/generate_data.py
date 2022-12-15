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
from Evaluation.average_face import average_face
from Evaluation.fpage.fpage import FPAge
import torch.nn.functional as F

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
    generate_image_folder: bool,
    generate_average_face: bool,
    make_truncation_data: bool,
    make_id_vs_age: bool,
    make_fancy_age: bool,
    samples_per_age: int,
    make_angles_data: bool,
    make_scatter_data: bool,
    generate_id_data: bool,
    ):
    ## LOADING NETWORK ##
    print(f'Loading networks from "{network_folder}"...')
    device = torch.device('cuda')
    

    print("Loading network named:", network_pkl)

    with dnnlib.util.open_url(network_pkl_path) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    np.random.seed(seed) # setting the seed to get consistent results
    
    if generate_average_face:
        print("Making average faces of different ages...")
        average_face(G, save_name, device, truncation_psi, truncation_cutoff, get_camera_parameters, get_conditioning_parameter, image_grid, network_folder)
    if make_angles_data:
        print("Generating age data...")
        generate_angles_data(G, age_model_name, device, angles_p, angles_y, angles_plot_iterations, save_name, truncation_cutoff, truncation_psi, network_folder)
    if generate_id_data:
        print("Generating id data...")
        ages_id = generate_id_data(G, device, id_plot_iterations, save_name, truncation_cutoff, truncation_psi, network_folder)
    if make_scatter_data:
        print("Generating scatter plot data...")
        generate_scatter_data(G, device, seed, save_name, network_folder, truncation_cutoff, truncation_psi, age_model_name, scatter_iterations, make_id_vs_age=make_id_vs_age)
    if make_truncation_data:
        print("Generating truncation data...")
        generate_truncation_data(G, device, seed, save_name, network_folder)
    print("Generating evaluation image...")
    generate_image(G, seed, device, network_folder, save_name, truncation_psi)
    if make_fancy_age:
        print('Generating fancy age scatter plot data')
        generate_fancy_age(G, device, seed, save_name, network_folder, age_model_name, truncation_psi, truncation_cutoff, samples_per_age=samples_per_age)
    if generate_image_folder:
        print("Generating image folders...")
        save_image_folder(save_name, network_folder, G, device, truncation_cutoff, truncation_psi)
    del G

def generate_scatter_data(G, device, seed, save_name, network_folder, truncation_cutoff, truncation_psi, age_model_name, iterations, make_id_vs_age=True):
    ## Age evaluation
    magface = FaceIDLoss(device, model="MagFace")
    FPAge_model = FPAge()
    training_option_path = os.path.join(network_folder, "training_options.json")
    f = open(training_option_path)
    training_option = json.load(f)
    f.close()
    age_loss_fn = training_option['age_loss_fn']
    age_min = training_option['age_min']
    age_max = training_option['age_max']

    if age_model_name == 'coral':
        age_model = Coral()
    elif age_model_name == 'v1':
        age_model = AgeEstimator()
    elif age_model_name == 'v2':
        age_model = AgeEstimatorNew(device, age_min = age_min, age_max=age_max)


    # angles_p = np.random.RandomState(seed).uniform(-0.5, 0.5, size=(iterations))
    # angles_y = np.random.RandomState(seed+1).uniform(-0.5, 0.5, size=(iterations))
    z = torch.from_numpy(np.random.RandomState(seed).randn(iterations, G.z_dim)).to(device)

    if age_loss_fn == "CAT":
        categories = 101
        ages = np.zeros((iterations, categories))
        ages_2 = np.zeros((iterations, categories))
        for i, _ in enumerate(ages):
            age = np.random.randint(age_min, age_max+1)
            ages[i][age] = 1
            age_2 = np.random.randint(age_min, age_max+1)
            ages_2[i][age] = 1
    elif age_loss_fn == "MSE":
        ages = np.random.RandomState(seed+2).uniform(-1, 1, size=(iterations, 1))
        ages_2 = np.random.RandomState(seed+3).uniform(-1, 1, size=(iterations, 1))

    fov_deg = 18.837
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0,0,0]), device=device)
    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
    res = []

    training_id_model = training_option['loss_kwargs']['id_model']
    
    id_model = FaceIDLoss(device, model = training_id_model)

    arcface = FaceIDLoss(device, model="ArcFace")
        
    cosine_sim_f = torch.nn.CosineSimilarity()
    for zi, age, age_2, angle_p, angle_y in tqdm(zip(z, ages, ages_2, [0]*iterations, [0]*iterations), total=iterations): #frontal view
        cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
        conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)    
        c = torch.cat((conditioning_params, torch.tensor(np.array([age]), device=device)), 1)
        c_params = torch.cat((camera_params, torch.tensor(np.array([age]), device=device)), 1).float()
        ws = G.mapping(zi[None,:], c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        img = G.synthesis(ws, c_params)['image']
        age_hat, _ = age_model.estimate_age(img)
        age_hat2 = FPAge_model.estimate_age(img)

        # same person, diff age
        if make_id_vs_age:
            c_params_2 = torch.cat((camera_params, torch.tensor(np.array([age_2]), device=device)), 1).float()
            c_2 = torch.cat((conditioning_params, torch.tensor(np.array([age_2]), device=device)), 1)
            ws_2 = G.mapping(zi[None,:], c_2, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
            img_2 = G.synthesis(ws_2, c_params_2)['image']
            # age_hat_2, logits_2 = age_model.estimate_age(img_2)
        
            v1 = id_model.get_feature_vector(img)
            v2 = id_model.get_feature_vector(img_2)
            cos_sim = cosine_sim_f(v1,v2)
            age_diff = abs(normalize(age, rmin=-1, rmax=1, tmin=age_min, tmax=age_max) - normalize(age_2, rmin=-1, rmax=1, tmin=age_min, tmax=age_max))

        if training_id_model != "MagFace":
            v1 = magface.get_feature_vector(img)
        mag = np.linalg.norm(v1.cpu().numpy())
        if age_loss_fn == "CAT":
            age_true = list(age).index(1)
            age_true = normalize(age_true, rmin=age_min, rmax=age_max)
        elif age_loss_fn =="MSE":
            age_true = age[0]
        if make_id_vs_age:
            res.append([age_hat.item(), age_hat2, age_true, angle_p, angle_y, mag, cos_sim.item(), age_diff[0]])
        else:
            res.append([age_hat.item(), age_hat2, age_true, angle_p, angle_y, mag])

    if make_id_vs_age:
        columns = ["age_hat", "age_hat2", "age_true", "angle_p", "angle_y", "mag", "cos_sim", "age_diff"]
    else:
        columns = ["age_hat", "age_hat2", "age_true", "angle_p", "angle_y", "mag"]

    df = pd.DataFrame(res, columns=columns)
    # Save as csv file
    save_dir = os.path.join("Evaluation","Runs", save_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "age_scatter.csv")
    print("Saving csv at", save_dir,"...")
    df.to_csv(save_path, index=False)   

def generate_id_data(
        G, device, id_plot_iterations, save_name, truncation_cutoff, truncation_psi, network_folder
    ):
    seeds = np.random.randint(1,100000, size = id_plot_iterations)

    training_option_path = os.path.join(network_folder, "training_options.json")
    f = open(training_option_path)
    training_option = json.load(f)
    f.close()
    training_id_model = training_option['loss_kwargs']['id_model']
    cosine_sim_f = torch.nn.CosineSimilarity()

    age_min = training_option['age_min']
    age_max = training_option['age_max']

    age_loss_fn = training_option['age_loss_fn']
    ages = list(map(int, np.linspace(age_min, age_max, 5)))
    id_model = FaceIDLoss(device, model=training_id_model)
    arcface = FaceIDLoss(device, model="ArcFace")
    res = []
    for seed in tqdm(seeds):
        for age1 in ages:
            angle_y = 0 # frontal view !np.random.uniform(-0.5,0.5)
            angle_p = 0 # frontal view !np.random.uniform(-0.5,0.5)
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
            c = get_conditioning_parameter(age1, G, device, age_loss_fn, age_min, age_max)
            c_camera = get_camera_parameters(age1, G, device, angle_y, angle_p, age_loss_fn, age_min, age_max)
            ws = G.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
            generated_image_1 =  G.synthesis(ws, c_camera)['image']
            feature_v_1 = id_model.get_feature_vector(generated_image_1)
            feature_v_1_arcface = arcface.get_feature_vector_arcface(generated_image_1)
            for age2 in ages:
                if age1 == age2:
                    pass # skip comparing similar images
                c = get_conditioning_parameter(age2, G, device, age_loss_fn, age_min, age_max)
                c_camera = get_camera_parameters(age2, G, device, angle_y, angle_p, age_loss_fn, age_min, age_max)
                ws = G.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
                generated_image_2 =  G.synthesis(ws, c_camera)['image']
                feature_v_2 = id_model.get_feature_vector(generated_image_2)
                feature_v_2_arcface = arcface.get_feature_vector_arcface(generated_image_2)
                cosine_sim = cosine_sim_f(feature_v_1, feature_v_2)
                cosine_sim_arcface = cosine_sim_f(feature_v_1_arcface, feature_v_2_arcface)
                res.append([seed, age1, age2, angle_y, angle_p, cosine_sim.item(), cosine_sim_arcface.item(), training_id_model])

    # create dataframe
    columns = ["seed", "age1", "age2", "angle_y", "angle_p", "cosine_sim", "cosine_sim_arcface", "id_train"]
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
        C3AE_model
    """
    seeds = np.random.randint(1,100000, size = angles_plot_iterations)
    training_option_path = os.path.join(network_folder, "training_options.json")
    f = open(training_option_path)
    training_option = json.load(f)
    f.close()
    age_min = training_option['age_min']
    age_max = training_option['age_max']
    age_loss_fn = training_option['age_loss_fn']
    magface = FaceIDLoss(device, model="MagFace")
    res = []
    batch_size = 8
    iterations_per_batch = 4
    n = 400
    iterations = n // batch_size
    # cs = []; cs_cameras = []; ages = []; angles_y = []; angles_p = []
    angles = []
    for angle_p in np.linspace(-0.8,0.8,9):
        for angle_y in np.linspace(-0.8,0.8,9):
            angles.append((angle_p, angle_y))
    for angle in tqdm(angles):
        for i in range(iterations_per_batch):
            z = torch.from_numpy(np.random.randn(batch_size, G.z_dim)).to(device)
            cs = []; cs_cameras = []; ages = []; angles_y = []; angles_p = []
            for i in range(batch_size):
                age = np.random.randint(age_min, age_max + 1)
                ages.append(age)
                angle_p, angle_y = angle
                angles_y.append(angle_y)
                angles_p.append(angle_p)
                cs.append(get_conditioning_parameter(age, G, device, age_loss_fn, age_min, age_max)[0])
                cs_cameras.append(get_camera_parameters(age, G, device, angle_y, angle_p, age_loss_fn, age_min, age_max)[0])
            c = torch.stack(cs)
            c_camera = torch.stack(cs_cameras)
            ws = G.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=14)
            generated_image =  G.synthesis(ws, c_camera)['image']
            features =  magface.get_feature_vector(generated_image)
            mag = np.linalg.norm(features.cpu().numpy(), axis=1)
            for i in range(batch_size):
                res.append([ages[i], angles_y[i], angles_p[i], mag[i]])
    # create dataframe
    columns = ["age", "angle_y", "angle_p", "magface"]
    df = pd.DataFrame(res, columns=columns)

    # Save as csv file
    save_dir = os.path.join("Evaluation", "Runs", save_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir,f"age_evaluation.csv")
    print("Saving csv at", save_dir,"...")
    df.to_csv(save_path, index=False)

def generate_image(G, seed, device, path, save_name, truncation_psi):
    magface = FaceIDLoss(device, model="MagFace")
    training_option_path = os.path.join(path, "training_options.json")
    f = open(training_option_path)
    training_option = json.load(f)
    f.close()
    age_loss_fn = training_option['age_loss_fn']
    age_min = training_option['age_min']
    age_max = training_option['age_max']
    angles_y = [0.5, 0, -0.5]
    angles_p = [0.5, 0.25, 0, -0.25, -0.5]
    fov_deg = 18.837

    intrinsics = FOV_to_intrinsics(fov_deg, device=device)

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
                c = torch.cat((conditioning_params, torch.tensor(np.array([age]), device=device)), 1)
                c_params = torch.cat((camera_params, torch.tensor(np.array([age]), device=device)), 1).float()

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
    save_path = os.path.join(save_dir, "image_random_age_set_angles.png")
    print("Saving image at", save_dir,"...")

    grid = image_grid(images, rows=len(angles_p), cols = images_per_angle*len(angles_y))
    grid.save(save_path)

    ## Image of linearly spread ages
    images = []
    ages = np.linspace(age_min, age_max, 8).round(0)
    #frontal view
    cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + 0, np.pi/2 + 0, cam_pivot, radius=cam_radius, device=device)
    conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
    camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1) 
    for age_g in ages:
        z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)
        if age_loss_fn == "CAT":
            age = [0] * 101
            age[age_g] = 1
        elif age_loss_fn == "MSE":
            age = [normalize(age_g, rmin=age_min, rmax=age_max)]

        c = torch.cat((conditioning_params, torch.tensor(np.array([age]), device=device)), 1)
        c_params = torch.cat((camera_params, torch.tensor(np.array([age]), device=device)), 1).float()

        ws = G.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        img = G.synthesis(ws, c_params)['image']
        features =  magface.get_feature_vector(img)
        mag = np.linalg.norm(features.cpu().numpy(), axis=1)
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)

        pil_img = Image.fromarray(img.permute(0,2,3,1)[0].cpu().numpy(), 'RGB')
        text_added = ImageDraw.Draw(pil_img)
        
        text_added.text((0,420), f"Age: {age_g} + {int(mag[0])}", font=font, fill=text_color)
        images.append(pil_img)

    save_dir = os.path.join("Evaluation","Runs", save_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "image_ages_frontal.png")
    print("Saving image at", save_dir,"...")

    grid = image_grid(images, rows = 2, cols = 4)
    grid.save(save_path)

    ### many images in one grid
    images=[]
    rows = 10
    columns = 8

    for i in range(int(rows * columns)):
        z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)
        age = [np.random.uniform(-1,1)]
        c = torch.cat((conditioning_params, torch.tensor(np.array([age]), device=device)), 1)
        c_params = torch.cat((camera_params, torch.tensor(np.array([age]), device=device)), 1).float()

        ws = G.mapping(z, c, truncation_psi=1, truncation_cutoff=truncation_cutoff)
        img = G.synthesis(ws, c_params)['image']
        img = (img * 127.5 + 128).clamp(0, 255)
        img = F.interpolate(img.float(), [150,150],  mode='bilinear', align_corners=True).to(torch.uint8)
        pil_img = Image.fromarray(img.permute(0,2,3,1)[0].cpu().numpy(), 'RGB')
        images.append(pil_img)
        
    save_path = os.path.join(save_dir, "image_many.png")
    print("Saving image at", save_dir,"...")

    grid = image_grid(images, rows = rows, cols = columns)
    grid.save(save_path)

    # used for evaluating magface score
    # font = ImageFont.truetype("FreeSerif.ttf", 40) 
    # # <400, 400-500, 500-600, 600-700, 700-800, 800-900, 900-1000, >1000
    # magface_buckets = [False] * 8
    # images = [0] * 8
    # while True:
    #     if sum(magface_buckets) == 8:
    #         break
    #     age_g = np.random.randint(age_min, age_max + 1)
    #     if age_loss_fn == "CAT":
    #         age = [0] * 101
    #         age[age_g] = 1
    #     elif age_loss_fn == "MSE":
    #         age = [normalize(age_g, rmin=age_min, rmax=age_max)]
    #     z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)
    #     c = torch.cat((conditioning_params, torch.tensor(np.array([age]), device=device)), 1)
    #     c_params = torch.cat((camera_params, torch.tensor(np.array([age]), device=device)), 1).float()
    #     ws = G.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
    #     img = G.synthesis(ws, c_params)['image']

    #     features =  magface.get_feature_vector(img)
    #     mag = np.linalg.norm(features.cpu().numpy(), axis=1)
        
    #     if (mag < 400):
    #         if (not magface_buckets[0]):
    #             magface_buckets[0] = True
    #             i = 0
    #         else:
    #             continue
    #     elif (mag < 500):
    #         if (not magface_buckets[1]):
    #             magface_buckets[1] = True
    #             i = 1
    #         else:
    #             continue
    #     elif (mag < 600):
    #         if (not magface_buckets[2]):
    #             magface_buckets[2] = True
    #             i = 2
    #         else:
    #             continue
    #     elif (mag < 700):
    #         if (not magface_buckets[3]):
    #             magface_buckets[3] = True
    #             i =3
    #         else:
    #             continue
    #     elif (mag < 800):
    #         if (not magface_buckets[4]):
    #             magface_buckets[4] = True
    #             i = 4
    #         else:
    #             continue
    #     elif (mag < 900):
    #         if (not magface_buckets[5]):
    #             magface_buckets[5] = True
    #             i = 5
    #         else:
    #             continue
    #     elif (mag < 1000):
    #         if (not magface_buckets[6]):
    #             magface_buckets[6] = True
    #             i = 6
    #         else:
    #             continue
    #     elif (mag >= 1000):
    #         if (not magface_buckets[7]):
    #             magface_buckets[7] = True
    #             i = 7
    #         else:
    #             continue
        

    #     img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    #     pil_img = Image.fromarray(img.permute(0,2,3,1)[0].cpu().numpy(), 'RGB')
    #     text_added = ImageDraw.Draw(pil_img)

    #     text_added.text((0,512-45), f"Magface score: {int(mag[0])}", font=font, fill=text_color)
    #     images[i] = pil_img

    # save_dir = os.path.join("Evaluation","Runs", save_name)
    # os.makedirs(save_dir, exist_ok=True)
    # save_path = os.path.join(save_dir, "image_magface.png")
    # print("Saving image at", save_dir,"...")

    # grid = image_grid(images, rows = 2, cols = 4)
    # grid.save(save_path)

def generate_fancy_age(G, device, seed, save_name, network_folder, age_model_name, truncation_psi, truncation_cutoff, samples_per_age=20):
    training_option_path = os.path.join(network_folder, "training_options.json")
    f = open(training_option_path)
    training_option = json.load(f)
    f.close()
    age_loss_fn = training_option['age_loss_fn']
    age_min = training_option['age_min']
    age_max = training_option['age_max']
    FPAge_model = FPAge()
    if age_model_name == 'coral':
        age_model = Coral()
    elif age_model_name == 'v1':
        age_model = AgeEstimator()
    elif age_model_name == 'v2':
        age_model = AgeEstimatorNew(device, age_min = age_min, age_max=age_max)

    fov_deg = 18.837
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)

    angle_y, angle_p = (0, 0) #frontal view
    cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
    cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
    conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
    camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

    ages = list(range(age_min, age_max + 5, 5))
    res = []

    for i, age in enumerate(tqdm(ages)):
        norm_age = normalize(age, rmin=age_min, rmax=age_max)
        c = torch.cat((conditioning_params, torch.tensor([[norm_age]], device=device)), 1)
        c_params = torch.cat((camera_params, torch.tensor([[norm_age]], device=device)), 1)
        c_params = c_params.float()
        zs = torch.from_numpy(np.random.RandomState(seed + i).randn(samples_per_age, G.z_dim)).to(device)
        age_hats = []; age_hats_fpage = []
        for z in zs:
            ws = G.mapping(z[None,:], c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
            img = G.synthesis(ws, c_params)['image']
            age_hat, _ = age_model.estimate_age(img, normalize=False)
            age_hat_fpage = FPAge_model.estimate_age(img)
            age_hats += [age_hat.item()]
            age_hats_fpage.append(age_hat_fpage)
        age_hats = np.array(age_hats)
        age_hats_fpage = np.array(age_hats_fpage)
        res.append([age, np.mean(np.abs(age_hats - age)), np.std(np.abs(age_hats) - age), np.mean(np.abs(age_hats_fpage - age)), np.std(np.abs(age_hats_fpage) - age)])

    df = pd.DataFrame(res, columns=['target_age', 'age_difference', 'std', 'age_difference_fpage', 'std_fpage'])

    save_dir = os.path.join("Evaluation", "Runs", save_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir,f"fancy_scatter.csv")
    print("Saving csv at", save_dir,"...")
    df.to_csv(save_path, index=False)


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
        l[age] = 1
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
        angle_y (float): yaw of synthetic person
        angle_p (float): pitch of synthetic person
        fov_deg (float, optional): Field of view. Defaults to 18.837.
    Returns:
        tensor: camera parameters
    """
    if age_loss_fn == "MSE":
        age_c = [normalize(age, rmin=age_min, rmax=age_max)]
    elif age_loss_fn == "CAT":
        l = [0] * 101
        l[age] = 1
        age_c = np.array(l)

    intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
    cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
    camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    camera_params = torch.cat((camera_params, torch.tensor(np.array([age_c]), device=device)), 1)
    return camera_params.float()

def generate_truncation_data(G, device, seed, save_name, network_folder):

    training_option_path = os.path.join(network_folder, "training_options.json")
    f = open(training_option_path)
    training_option = json.load(f)
    f.close()
    age_min = training_option['age_min']
    age_max = training_option['age_max']
    age_loss_fn = training_option['age_loss_fn']
    magface = FaceIDLoss(device, model="MagFace")
    age_model = AgeEstimatorNew(device, age_min=age_min, age_max=age_max)
    images_per_trunc_value = 96//16
    trunc_values = 17
    number_of_images = images_per_trunc_value * trunc_values
    truncations = np.linspace(0.2,1,trunc_values).repeat(images_per_trunc_value)
    seeds = np.random.RandomState(seed).randint(1,100000, size = number_of_images)
    ages = np.random.RandomState(seed+2).randint(age_min, age_max, size=number_of_images)
    res=[]
    batch_size = 16
    for truncation_psi in tqdm(truncations):
        
        z = torch.from_numpy(np.random.randn(batch_size, G.z_dim)).to(device)
        cs = []
        cs_cameras = []
        ages = []
        angle_y, angle_p = 0,0
        for i in range(batch_size):
            age = np.random.randint(age_min, age_max + 1)
            ages.append(age)
            cs.append(get_conditioning_parameter(age, G, device, age_loss_fn, age_min, age_max)[0])
            cs_cameras.append(get_camera_parameters(age, G, device, angle_y, angle_p, age_loss_fn, age_min, age_max)[0])
        c = torch.stack(cs)
        c_camera = torch.stack(cs_cameras)
        ws = G.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=14)
        generated_image =  G.synthesis(ws, c_camera)['image']
        features =  magface.get_feature_vector(generated_image)
        mag = np.linalg.norm(features.cpu().numpy(), axis=1)
        age_hat, _ = age_model.estimate_age(generated_image)
        for i in range(batch_size):
            res.append([ages[i], angle_y, angle_p, age_hat[i].item(), truncation_psi, mag[i]])

    # create dataframe
    columns = ["age", "angle_y", "angle_p", "age_hat", "truncation_psi", "mag"]
    df = pd.DataFrame(res, columns=columns)

    # Save as csv file
    save_dir = os.path.join("Evaluation", "Runs", save_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir,f"truncation_data.csv")
    print("Saving csv at", save_dir,"...")
    df.to_csv(save_path, index=False) 

def save_image_folder(save_name, network_folder, G, device, truncation_cutoff, truncation_psi):
    seed=0
    training_option_path = os.path.join(network_folder, "training_options.json")
    f = open(training_option_path)
    training_option = json.load(f)
    f.close()
    age_loss_fn = training_option['age_loss_fn']
    age_min = training_option['age_min']
    age_max = training_option['age_max']
    number_of_images = 400
    
    z = torch.from_numpy(np.random.RandomState(seed).randn(number_of_images, G.z_dim)).to(device)
    magface = FaceIDLoss(device, model="MagFace")
    # if age_loss_fn == "CAT":
    #     categories = 101
    #     ages = np.zeros((iterations, categories))
    #     for i, _ in enumerate(ages):
    #         age = np.random.randint(age_min, age_max+1)
    #         ages[i][age] = 1
    if age_loss_fn == "MSE":
        ages = np.random.RandomState(seed+2).uniform(-1, 1, size=(number_of_images, 1))

    fov_deg = 18.837
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0,0,0]), device=device)
    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
    folder = os.path.join("Evaluation/Runs", save_name, "random_angles")
    os.makedirs(folder, exist_ok=True)

    age_model = AgeEstimatorNew(device, age_min = age_min, age_max=age_max)

    folder = os.path.join("Evaluation/Runs", save_name, "no_angles")
    os.makedirs(folder, exist_ok=True)
    z = torch.from_numpy(np.random.RandomState(seed).randn(number_of_images, G.z_dim)).to(device)
    for zi, age in tqdm(zip(z, ages), total=number_of_images):
        angle_p, angle_y = 0,0
        cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
        conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)    
        c = torch.cat((conditioning_params, torch.tensor(np.array([age]), device=device)), 1)
        c_params = torch.cat((camera_params, torch.tensor(np.array([age]), device=device)), 1).float()

        ws = G.mapping(zi[None,:], c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        img = G.synthesis(ws, c_params)['image']
        age_hat,_ = age_model.estimate_age(img, normalize=False)
        f = magface.get_feature_vector(img)
        mag = np.linalg.norm(f.cpu().numpy())
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        image_name = f"{denormalize(age, rmin=age_min, rmax=age_max).round(2)[0]}-{round(age_hat.item(),1)}-{truncation_psi}-{mag.round(1)}".replace(".",",")
        pil_img = Image.fromarray(img[0].detach().cpu().numpy(), 'RGB')
        pil_img.save(os.path.join(folder, image_name) + ".png")

if __name__ == "__main__":
    generate_data()