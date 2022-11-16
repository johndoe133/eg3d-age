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

def plot_fancy_age(network_folder, save_name):
    plot_setup()
    fig_name='fancy_scatter_plot'
    training_option_path = os.path.join(network_folder, "training_options.json")
    f = open(training_option_path)
    training_option = json.load(f)
    if 'age_min' in training_option.keys():
        age_min = training_option['age_min']
        age_max = training_option['age_max']
    else:
        age_min = 0
        age_max = 100
    save_path = os.path.join("Evaluation", "Runs", save_name)
    df = pd.read_csv(os.path.join(save_path, "fancy_scatter.csv"))

    figsize = compute_figsize(350, 250)
    plt.figure(figsize=figsize, dpi=300)
    plt.xticks(list(range(age_min, age_max + 5, 5)))
    
    
    plt.scatter(df['target_age'], df['age_difference'], s=10, zorder=20)
    plt.plot(df['target_age'], df['age_difference'])
    plt.fill_between(df['target_age'], df['age_difference'] + df['std'], df['age_difference'] - df['std'], facecolor='blue', alpha=0.4)
    lower, upper = plt.ylim()
    plt.ylim(0, upper)
    plt.xlabel('Target age')
    plt.ylabel('Age difference')


    plt.savefig(save_path + f"/{fig_name}" + ".png",bbox_inches='tight')
    plt.savefig(save_path + f"/{fig_name}" + ".pgf",bbox_inches='tight')
