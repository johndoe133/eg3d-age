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
from Evaluation.numbers import save_correlation


@click.command()
@click.option('--network_folder', help='Network folder name', required=True)
@click.option('--network', help='Network folder name', default=None, required=False)
@click.option('--seed', help='Seed to generate from', default=42, required=False, type=int)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--angles_plot_iterations', help='Number of random seeds to generate synthetic images from when making the angles plot', default=10, type=int)
@click.option('--id_plot_iterations', help='Number of random seeds to generate synthetic images from when making the id plot', default=10, type=int)
@click.option('--age_model_name', help='Age model used', default="DEX", type=str)
@click.option('--angles_p', help='', cls=PythonLiteralOption, required=False, default="[0.4, 0, -0.4]")
@click.option('--angles_y', help='', cls=PythonLiteralOption, required=False, default="[0.4, 0, -0.4]")
@click.option('--run_generate_data', help='',required=False, default=True)
@click.option('--scatter_iterations', help='Number of generated random synthetic faces for scatter plot',required=False, default=100)
@click.option('--numbers', help='Make csv of hard number evaluations like correlation', required=False, is_flag=True, default=True)

def run_evaluation(
    network_folder: str,
    network: str,
    seed: int,
    truncation_psi: float,
    truncation_cutoff: int,
    angles_plot_iterations: int,
    id_plot_iterations: int, 
    age_model_name: str,
    angles_p: list,
    angles_y: list,
    run_generate_data: bool,
    scatter_iterations: int,
    numbers: bool,
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

    print("Generating data...")
    if run_generate_data:
        ages, ages_id = generate_data(save_name, network_folder, network_pkl_path, network_pkl, seed, truncation_psi, truncation_cutoff, angles_plot_iterations, age_model_name, angles_p, angles_y, scatter_iterations, id_plot_iterations)
    
    print("Creating plots...")
    scatter_plot(network_folder, save_name)

    angles_plot(save_name, angles_p, angles_y)

    set_age_plot(save_name, ages)

    id_plot(save_name, ages_id)

    save_correlation(save_name, network_folder=network_folder)

    
if __name__=='__main__':
    run_evaluation()