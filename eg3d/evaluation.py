import click 
import torch
import dnnlib
import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from training.estimate_age import AgeEstimator
from training.training_loop import normalize, denormalize
from tqdm import tqdm
from training.face_id import FaceIDLoss
from scipy.stats import gaussian_kde
from training.coral import Coral
from plot_training_results import plot_setup, compute_figsize

@click.command()
@click.option('--network_folder', help='Network folder name', required=True)
@click.option('--network', help='Network folder name', default=None, required=False)
@click.option('--seed', help='Seed to generate from', default=42, required=False, type=int)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--create_graph', help="Whether to generate a graph", default=True, type=bool)
@click.option('--run_eval', help='Whether to run evaluation loop', default=True, type=bool)
@click.option('--no-img', help='Number of random seeds to generate synthetic images from', default=10, type=int)
@click.option('--model_name', help='Age model used', default="coral", type=str)

def evaluate(
    network_folder: str,
    network: str,
    seed: int,
    truncation_psi: float,
    truncation_cutoff: int,
    create_graph: bool,
    run_eval: bool,
    no_img: int,
    model_name: str,

    ):
    ## LOADING NETWORK ##
    print(f'Loading networks from "{network_folder}"...')
    device = torch.device('cuda')
    seeds = np.random.randint(1,100000, size=no_img)
    if network is not None: # choose specific network
        network_pkl = network
    else: # choose the network trained the longest
        pkls = [string for string in os.listdir(network_folder) if '.pkl' in string]
        pkls = sorted(pkls)
        network_pkl = pkls[-1]
    
    ages = np.round(np.linspace(16,70,9))
    ages_id = np.linspace(5,80,6)
    angles_p = [0.3, 0, -0.3]
    angles_y = [.4, 0, -.4]
    if run_eval:
        run_age_evaluation(model_name, ages, angles_p, angles_y, network_folder, network_pkl, seed, device, truncation_cutoff, truncation_psi, seeds)
        run_id_evaluation(ages_id, network_folder, network_pkl, seed, device, truncation_cutoff, truncation_psi, seeds)
    
    save_dir = os.path.join("Evaluations", network_folder.split("/")[-1])
    save_path_age = os.path.join(save_dir, "age_evaluation.csv")
    save_path_id = os.path.join(save_dir, "id_evaluation.csv")

    if create_graph:
        generate_age_plot(save_path_age, save_dir, ages, angles_p, angles_y)
        generate_id_plot(save_path_id ,save_dir, ages_id)


global save_path, save_dir

def get_conditioning_parameter(age, G, device, fov_deg = 18.837):
    """Get conditioning parameters for a given age, looking at the camera

    Args:
        age (int)
        G : generator

        device
        fov_deg (float, optional): Field of view. Defaults to 18.837.

    Returns:
        tensor: conditioning parameters
    """
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
    conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
    conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    conditioning_params = torch.cat((conditioning_params, torch.tensor([[normalize(age)]], device=device)), 1) # add age
    return conditioning_params.float()

def get_camera_parameters(age, G, device, angle_y, angle_p, fov_deg = 18.837):
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
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
    cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
    camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    camera_params = torch.cat((camera_params, torch.tensor([[normalize(age)]], device=device)), 1)
    return camera_params.float()

def estimate_age(img, age):
    return np.random.randint(age-5,age+5)

def get_feature_vector(img):
    return np.random.randint(0,1,size=10)
# Evaluation pipeline

def generate_age_plot(save_path,save_dir, ages, angles_p, angles_y):
    plot_setup()
    df = pd.read_csv(save_path)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    figsize = compute_figsize(426, 500)
    fig, axs = plt.subplots(3, 1, sharex=False, figsize = figsize, dpi=300)
    for i, age in enumerate(ages):
        df_age = df[df.age==age]
        age_hat = df_age.age_hat.to_numpy()
        true_age = df_age.age.to_numpy()
        age_hat = age_hat[:, np.newaxis]

        #Density plot
        kde = KernelDensity(kernel="gaussian", bandwidth=0.5).fit(age_hat)
        x = np.linspace(-5,90, 1000)[:, np.newaxis]
        log_dens = kde.score_samples(x)
        axs[0].fill(x[:, 0], np.exp(log_dens), alpha=0.5, label=f"Age: {age}")
        
        # axs[1].hist(age_hat, bins=np.linspace(5,85,200))

        # axs[2].scatter(true_age, age_hat, label=str(int(age)), facecolors='none', edgecolor=colors[i])
        # axs[2].set_xlabel("Desired age")
        # axs[2].set_ylabel("Predicted age")
    axs[0].set_xlim(-1,88)
    print("min",df.age_hat.min())
    print("max",df.age_hat.max())
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    
    plt.tight_layout()

    fig_name = "plot.png"
    plt.savefig(save_dir + f"/{fig_name}")
    print(f"Figure {fig_name} save at {save_dir}")

    fig, axs = plt.subplots(3, 3, sharex=True, sharey=True, figsize = figsize, dpi=300)

    pad = 5 # in points

    for ax, angle in zip(axs[0], angles_y):
        text = f"$Angle_y$={angle}"
        ax.annotate(text, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

    for ax, angle in zip(axs[:,0], angles_p):
        text = f"$Angle_p$={angle}"
        ax.annotate(text, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')
        
    for i, angle_y in enumerate(angles_y):
        for j, angle_p in enumerate(angles_p):
            df_angle = df[(df.angle_y==angle_y) & (df.angle_p==angle_p)]
            error = df_angle.error.to_numpy()
            # axs[i,j].hist(error, np.linspace(error.min(), error.max(), 20))
            error = error[:, np.newaxis]
            kde = KernelDensity(kernel="gaussian", bandwidth=1).fit(error)
            x = np.linspace(error.min() * 1.1, error.max() * 1.1, 1000)[:, np.newaxis]
            log_dens = kde.score_samples(x)
            axs[i,j].fill(x[:, 0], np.exp(log_dens), alpha=0.8)
            # axs[i,j].fill_between(x[:,0], np.zeros(1000), np.exp(log_dens))
            
            # Axis labels
            if axs[i,j].get_subplotspec().is_first_col():
                axs[i,j].set_ylabel("Normalized Density")
            if axs[i,j].get_subplotspec().is_last_row():
                axs[i,j].set_xlabel("Error")
            # axs[i,j].legend()
    fig.tight_layout()
    fig_name = "angles.png"
    plt.savefig(save_dir + f"/{fig_name}")
    print(f"Figure {fig_name} save at {save_dir}")

    fig, axs = plt.subplots(3, 3, sharex=True, sharey=True, figsize = figsize, dpi=300)
    for i, age in enumerate(ages):
        df_age = df[df.age==age]
        age_hat = df_age.age_hat.to_numpy()
        true_age = df_age.age.to_numpy()
        age_hat = age_hat[:, np.newaxis]

        #Density plot
        kde = KernelDensity(kernel="gaussian", bandwidth=0.5).fit(age_hat)
        x = np.linspace(-5,90, 1000)[:, np.newaxis]
        log_dens = kde.score_samples(x)
        axs.ravel()[i].fill_between(x[:, 0], np.zeros(1000), np.exp(log_dens), color=colors[i], edgecolor=None, alpha=0.8)

        ylimmin, ylimmax = axs.ravel()[i].get_ylim()
        axs.ravel()[i].vlines(int(age), 0, 1, linestyles = '--', colors="black", label=f"Set age")
        axs.ravel()[i].set_ylim(ylimmin, 0.1)

        axs.ravel()[i].set_xticks(np.linspace(0, 80, 5))
         # Axis labels
        if axs.ravel()[i].get_subplotspec().is_first_col():
            axs.ravel()[i].set_ylabel("Normalized Density")
        if axs.ravel()[i].get_subplotspec().is_last_row():
            axs.ravel()[i].set_xlabel("Estimated age")
        axs.ravel()[i].legend()

    fig.tight_layout()
    fig_name = "grid.png"
    plt.savefig(save_dir + f"/{fig_name}")
    print(f"Figure {fig_name} save at {save_dir}")
    

def generate_id_plot(save_path_id ,save_dir, ages_id):
    plot_setup()
    figsize = compute_figsize(426, 500)
    #id plot
    bandwidth=.005
    df = pd.read_csv(save_path_id)
    xlim = df.cosine_sim.min()
    xlim = xlim - (xlim*0.05)
    ages = ages_id
    pad = 5 # in points
    fig, axs = plt.subplots(len(ages), len(ages), sharex=True, sharey=True, figsize = figsize, dpi=300)
    for i, age1 in enumerate(ages):
        df_age = df[df.age1==age1]
        for j, age2 in enumerate(ages):
            ## AXIS LABELS
            if axs[i,j].get_subplotspec().is_last_row():
                axs[i,j].set_xlabel("Cosine\nsimilarity")

            if axs[i,j].get_subplotspec().is_first_row():
                text = f"Age: {int(age2)}"
                axs[i,j].annotate(text, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

            if axs[i,j].get_subplotspec().is_first_col():
                axs[i,j].set_ylabel("Density")
                text = f"Age:\n{int(age1)}"
                axs[i,j].annotate(text, xy=(0, 0.5), xytext=(-axs[i,j].yaxis.labelpad - pad, 0),
                    xycoords=axs[i,j].yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')
            
            if age1==age2:
                # skip comparing same image
                continue

            cos = df_age[df_age.age2 == age2].cosine_sim.to_numpy()
            kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
            kde.fit(cos[:, None])
            x_d = np.linspace(xlim, 1, 200)
            logprob = kde.score_samples(x_d[:, None])
            pdf = np.exp(logprob)
            axs[i,j].plot(x_d, pdf)
            axs[i,j].fill_between(x_d, pdf, alpha=0.5)
            axs[i,j].set_xlim(xlim, 1)
            
    fig.tight_layout()
    plt.subplots_adjust(
                    wspace=0.1,
                    hspace=0.1)
                    
    fig_name = "id.png"
    plt.savefig(save_dir + f"/{fig_name}")
    print(f"Figure {fig_name} save at {save_dir}")     


def run_age_evaluation(
    model_name, ages, angles_p, angles_y, network_folder, network_pkl, seed, 
    device, truncation_cutoff, truncation_psi, seeds
    ):
    
    network_pkl_path = os.path.join(network_folder, network_pkl)
    print("Loading network named:", network_pkl)

    with dnnlib.util.open_url(network_pkl_path) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    np.random.seed(seed)

    ## Age evaluation
    if model_name == 'coral':
        age_model = Coral()
    elif model_name == 'DEX':
        age_model = AgeEstimator()
    angles = []
    for angle_p in angles_p:
        for angle_y in angles_y:
            angles.append((angle_y, angle_p))

    
    res = []
    for seed in tqdm(seeds):
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        for age in tqdm(ages):
            c = get_conditioning_parameter(age, G, device)
            for angle_y, angle_p in angles:
                c_camera = get_camera_parameters(age, G, device, angle_y, angle_p)
                ws = G.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
                generated_image =  G.synthesis(ws, c_camera)['image']
                age_hat = age_model.estimate_age(generated_image)
                if model_name == "DEX":
                    age_hat = denormalize(age_hat)
                age_hat = age_hat.item()
                mae = np.abs(age - age_hat)
                error = age-age_hat
                res.append([seed, age, angle_y, angle_p, age_hat, mae, error])
    
    # create dataframe
    columns = ["seed", "age", "angle_y", "angle_p", "age_hat", "mae", "error"]
    df = pd.DataFrame(res, columns=columns)

    # Save as csv file
    save_dir = os.path.join("Evaluations", network_folder.split("/")[-1])
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "age_evaluation.csv")
    print("Saving csv at", save_dir,"...")
    df.to_csv(save_path, index=False)
    del G

def run_id_evaluation(
        ages, network_folder, network_pkl, seed, device, truncation_cutoff, truncation_psi, seeds
    ):
    network_pkl_path = os.path.join(network_folder, network_pkl)
    print("Loading network named:", network_pkl)

    with dnnlib.util.open_url(network_pkl_path) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    np.random.seed(seed)
    cosine_sim_f = torch.nn.CosineSimilarity()
    id_model = FaceIDLoss(device)
    res = []
    for seed in tqdm(seeds):
        for age1 in ages:
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
            c = get_conditioning_parameter(age1, G, device)
            c_camera = get_camera_parameters(age1, G, device, 0, 0)
            ws = G.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
            generated_image_1 =  G.synthesis(ws, c_camera)['image']
            feature_v_1 = id_model.get_feature_vector(generated_image_1)
            for age2 in ages:
                if age1 == age2:
                    continue # skip comparing similar images
                c = get_conditioning_parameter(age2, G, device)
                c_camera = get_camera_parameters(age2, G, device, 0, 0)
                ws = G.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
                generated_image_2 =  G.synthesis(ws, c_camera)['image']
                feature_v_2 = id_model.get_feature_vector(generated_image_2)
                cosine_sim = cosine_sim_f(feature_v_1, feature_v_2)
                res.append([seed, age1, age2, cosine_sim.item()])

    # create dataframe
    columns = ["seed", "age1", "age2", "cosine_sim"]
    df = pd.DataFrame(res, columns=columns)

    # Save as csv file
    save_dir = os.path.join("Evaluations", network_folder.split("/")[-1])
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "id_evaluation.csv")
    print("Saving csv at", save_dir,"...")
    df.to_csv(save_path, index=False)
    del G



if __name__ == "__main__":
    evaluate()
    