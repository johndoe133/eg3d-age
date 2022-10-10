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
from training.estimate_age import AgeEstimator, AgeEstimatorNew
from training.training_loop import normalize, denormalize
from tqdm import tqdm
from training.face_id import FaceIDLoss
from scipy.stats import gaussian_kde
from training.coral import Coral
from plot_training_results import plot_setup, compute_figsize
from train import PythonLiteralOption

from Evaluation.generate_data import generate_data
from Evaluation.Plots.angles_plot import angles_plot
from Evaluation.Plots.set_age_plot import set_age_plot
from Evaluation.Plots.id_similarity_plot import id_plot
from Evaluation.Plots.scatter import scatter_plot

@click.command()
@click.option('--network_folder', help='Network folder name', required=True)
@click.option('--network', help='Network folder name', default=None, required=False)
@click.option('--seed', help='Seed to generate from', default=42, required=False, type=int)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--no-img', help='Number of random seeds to generate synthetic images from', default=10, type=int)
@click.option('--age_model_name', help='Age model used', default="DEX", type=str)
@click.option('--ages', help='', cls=PythonLiteralOption, required=False, default="[0,5,10,20,30,40,60,70,80]")
@click.option('--ages_id', help='', cls=PythonLiteralOption, required=False, default="[5,20,35,65,80]")
@click.option('--angles_p', help='', cls=PythonLiteralOption, required=False, default="[0.3, 0, -0.3]")
@click.option('--angles_y', help='', cls=PythonLiteralOption, required=False, default="[0.4, 0, -0.4]")
@click.option('--run_generate_data', help='',required=False, default=True)
def run_evaluation(
    network_folder: str,
    network: str,
    seed: int,
    truncation_psi: float,
    truncation_cutoff: int,
    no_img: int,
    age_model_name: str,
    ages: list,
    ages_id: list,
    angles_p: list,
    angles_y: list,
    run_generate_data: bool,
    ):
    np.seterr(all="ignore") # ignore numpy warnings

    if network is not None: # choose specific network
        network_pkl = network
    else: # choose the network trained the longest
        pkls = [string for string in os.listdir(network_folder) if '.pkl' in string]
        pkls = sorted(pkls)
        network_pkl = pkls[-1]
    network_pkl_path = os.path.join(network_folder, network_pkl)
    save_name = f"{network_pkl_path.split('/')[2]}-{network_pkl.split('.')[0][8:]}"

    if run_generate_data:
        generate_data(save_name, network_folder, network_pkl_path, network_pkl, seed,truncation_psi, truncation_cutoff, no_img,age_model_name, ages, ages_id, angles_p, angles_y)
    
    scatter_plot(network_folder, save_name)

    angles_plot(save_name, angles_p, angles_y)

    set_age_plot(save_name, ages)
    
if __name__=='__main__':
    run_evaluation()