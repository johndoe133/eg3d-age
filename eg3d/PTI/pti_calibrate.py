# Invert every image in /zhome/d7/6/127158/Documents/eg3d-age/dataset_preprocessing/ffhq/Deep3DFaceRecon_pytorch/test/crop

# Save everything in work3

# Create folder in Evaluation/Inversion/{network}/{image_name}/age0.png, age10.png

import os
import sys
import pickle
import numpy as np
from PIL import Image
import torch
from configs import paths_config, hyperparameters, global_config
from scripts.run_pti import run_PTI
import matplotlib.pyplot as plt
import sys
import json
home = os.path.expanduser('~')
sys.path.append(os.path.join(home, 'Documents/eg3d-age/eg3d/training'))
from estimate_age import AgeEstimatorNew
from tqdm import tqdm
import click

import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

@click.command()
@click.option('--model_path', help="Relative path to the model", required=True)
@click.option('--trunc', help="Truncation psi in inversion", required=False, default=1.0)
def calibrate(
        model_path: str,
        trunc: float,
    ):
    save_name = f"{model_path.split('/')[-3]}-{model_path.split('/')[-1].split('.')[0][8:]}-trunc-{trunc}"
    
    paths_config.checkpoints_dir = f'/work3/morbj/embeddings/{save_name}'
    device = torch.device('cuda:0')
    image_folder = '/zhome/d7/6/127158/Documents/eg3d-age/dataset_preprocessing/ffhq/Deep3DFaceRecon_pytorch/test/crop'
    home = os.path.expanduser('~')
    json_path = r"Documents/eg3d-age/dataset_preprocessing/ffhq/Deep3DFaceRecon_pytorch/checkpoints/pretrained/results/test/epoch_20_000000/dataset.json"
    c_path = os.path.join(home, json_path)
    paths_config.stylegan2_ada_ffhq = model_path
    # Invert every image
    yu4u = AgeEstimatorNew(device)
    # Generate image of ages [0,5,10,...,100]
    ages = np.arange(0,105,5)
    targets = []
    estimates = []
    for image_name in tqdm(os.listdir(image_folder)):

        image_name = image_name.strip('.png')

        # load PTI G and z pivot
        try:
            new_G, z_pivot= load_generator(image_name)
        except:
            print("Particalur image", image_name, "not inverted. Possibly deleted. Continue")
            continue

        # prepare c
        f = open(c_path)
        data = json.load(f)
        dictionary = dict(data['labels'])
        c = dictionary[image_name + ".jpg"]
        f.close()
        c.append(0) # changed later
        c = np.reshape(c, (1, 26))
        c = torch.FloatTensor(c).cuda()
        # generate images of different ages
        for age in np.random.uniform(0,101,size=4):
            new_c = c
            new_c[0][-1] = normalize(age, rmin=hyperparameters.age_min, rmax=hyperparameters.age_max)
            w_pivot = new_G.mapping(z_pivot, c, truncation_psi = trunc)
            new_image = new_G.synthesis(w_pivot, new_c)['image']
            estimate,_ = yu4u.estimate_age(new_image, normalize=False)
            targets.append(age)
            estimates.append(estimate.item())
        del new_G
    
    targets = np.array(targets)
    estimates = np.array(estimates)
    a, b = np.polyfit(targets, estimates,1)
    plt.figure(dpi=300, figsize=(6,3))
    ax = plt.subplot(1,2,1)
    plt.scatter(targets, estimates, s=10,facecolors='none', edgecolors='C0')
    label = "Best fit:\n$\hat{y}= $" + str(round(a,3)) + "$y+$" + str(round(b,1))
    plt.xlabel("Target age, $y$")
    plt.ylabel("Estimated age, $\hat{y}")
    xlim, ylim = plt.xlim(), plt.ylim()
    plt.plot(np.linspace(-10,110,10), a*np.linspace(-10,110,10) + b, color='r', label=label, zorder=20)
    plt.plot([-20,120], [-20, 120],'--',color='black', label="Perfect prediction")
    plt.xlim(xlim)
    plt.ylim(xlim)
    plt.title("Non-calibrated")
    plt.legend()
    # plt.tight_layout()


    # plt.subplot(1,2,2, sharey=ax)
    # plt.ylabel("Estimated age, $\hat{y}")
    # plt.scatter(targets, estimates, s=10,facecolors='none', edgecolors='C0')
    # plt.plot(np.linspace(-10,110,10), a*np.linspace(-10,110,10) + b, color='r', zorder=20)
    # plt.plot([-20,120], [-20, 120],'--',color='black')
    # plt.vlines(c, -10, 60, color='C1', label=f"Calibration example")
    # plt.hlines(60, -10, c, color='C1')
    # plt.xlim(xlim)
    # plt.ylim(xlim)
    # plt.legend()

    targets = []
    estimates = []

    cal = lambda age: (age-b)/a

    for image_name in tqdm(os.listdir(image_folder)):

        image_name = image_name.strip('.png')

        # load PTI G and z pivot
        try:
            new_G, z_pivot= load_generator(image_name)
        except:
            print("Particalur image", image_name, "not inverted. Possibly deleted. Continue")
            continue

        # prepare c
        f = open(c_path)
        data = json.load(f)
        dictionary = dict(data['labels'])
        c = dictionary[image_name + ".jpg"]
        f.close()
        c.append(0) # changed later
        c = np.reshape(c, (1, 26))
        c = torch.FloatTensor(c).cuda()
        # generate images of different ages
        for age in np.random.uniform(0,101,size=4):
            cal_age = cal(age)
            new_c = c
            new_c[0][-1] = normalize(cal_age, rmin=hyperparameters.age_min, rmax=hyperparameters.age_max)
            w_pivot = new_G.mapping(z_pivot, c, truncation_psi = trunc)
            new_image = new_G.synthesis(w_pivot, new_c)['image']
            estimate,_ = yu4u.estimate_age(new_image, normalize=False)
            targets.append(age)
            estimates.append(estimate.item())
        del new_G

    plt.subplot(1,2,2, sharey=ax)
    plt.xlabel("Target age, $y$")
    a, b = np.polyfit(targets, estimates,1)
    # plt.ylabel("Estimated age, $\hat{y}")
    plt.scatter(targets, estimates, s=10,facecolors='none', edgecolors='C0')
    plt.plot(np.linspace(-10,110,10), a*np.linspace(-10,110,10) + b, color='r', zorder=20, label="New best fit")
    plt.plot([-20,120], [-20, 120],'--',color='black', label="Perfect prediction")
    plt.xlim(xlim)
    plt.ylim(xlim)
    plt.title("Calibrated")
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join("Evaluation", "Runs", save_name)
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, "PTI_calibration.png"), bbox_inches="tight")
    plt.savefig(os.path.join(save_path, "PTI_calibration.pgf"), bbox_inches="tight")


def normalize(x, rmin = 0, rmax = 75, tmin = -1, tmax = 1):
    """Defined in eg3d.training.training_loop
    """
    z = ((x - rmin) / (rmax - rmin)) * (tmax - tmin) + tmin
    return np.round(z, 4)


def load_generator(image_name):
    with open(f'{paths_config.checkpoints_dir}/G/{image_name}.pt', 'rb') as f_new: 
        new_G = torch.load(f_new).cuda()
    z_pivot = torch.load(f'{paths_config.checkpoints_dir}/w/{image_name}.pt')
    return new_G, z_pivot
    
if __name__ == "__main__":
    calibrate()