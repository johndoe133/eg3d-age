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

    figsize = compute_figsize(400, 250)
    plt.figure(figsize=figsize, dpi=300)
    plt.xticks(list(range(age_min, age_max + 5, 5)))

    # yu4u
    ax = plt.subplot(1,2,1)
    plt.ylabel('Mean age difference')
    plt.xlabel('Target age')
    plt.scatter(df['target_age'], df['age_difference'], s=10, zorder=20, label='yu4u model')
    plt.plot(df['target_age'], df['age_difference'])
    plt.fill_between(df['target_age'], df['age_difference'] + df['std'], df['age_difference'] - df['std'], facecolor='C0', alpha=0.3, label='std')
    plt.gca().set_box_aspect(1)
    plt.xticks(np.arange(0,90,10))
    plt.title("yu4u model")
    #FPAge
    plt.subplot(1,2,2, sharey=ax)
    plt.scatter(df['target_age'], df['age_difference_fpage'], s=10, zorder=20, label='FPAge model', color='C1')
    plt.plot(df['target_age'], df['age_difference_fpage'], color='C1')
    plt.fill_between(df['target_age'], df['age_difference_fpage'] + df['std_fpage'], df['age_difference_fpage'] - df['std_fpage'], facecolor='C1', alpha=0.3, label='std')
    lower, upper = plt.ylim()
    plt.ylim(0, upper)
    plt.xlabel('Target age')
    plt.gca().set_box_aspect(1) 
    plt.xticks(np.arange(0,90,10))
    plt.title("FPAge model")
    plt.tight_layout()
    plt.savefig(save_path + f"/{fig_name}" + ".png",bbox_inches='tight')
    plt.savefig(save_path + f"/{fig_name}" + ".pgf",bbox_inches='tight')
    print(f"Saved {fig_name} at {save_path}...")

    if not os.path.isfile(os.path.join(save_path, "pti.csv")):
        print("Did not make PTI plot as data does not exist")
        return False
    
    df = pd.read_csv(os.path.join(save_path, "pti.csv"), index_col=0)
    df['error_yu4u'] = np.abs(df.target_age - df.age_hat_yu4u)
    df['error_fpage'] = np.abs(df.target_age - df.age_hat_fpage)
    figsize = compute_figsize(400, 450)
    plt.figure(figsize=figsize, dpi=300)
    df_mean_by_target = df.groupby(df.target_age).mean().reset_index()
    df_std_by_target = df.groupby(df.target_age).std().reset_index()

    ax = plt.subplot(2,1,1) # yu4u
    x = df_mean_by_target['target_age']
    y = df_mean_by_target['error_yu4u']
    std_error_yu4u = df_std_by_target.error_yu4u
    plt.scatter(x, y, s=10, zorder=20, label="Age-EG3D")
    plt.plot(x, y)
    plt.fill_between(x, y + std_error_yu4u, y - std_error_yu4u, facecolor='C0', alpha=0.3, label='std')
    # plt.gca().set_box_aspect(1)
    plt.xticks(np.arange(0,110,10))
    plt.title("yu4u model")
    plt.xlabel("Target age")
    plt.ylabel("MAE")
    plt.yticks(np.arange(0,plt.ylim()[1], 5))
    plt.ylim(0,plt.ylim()[1])
    plt.grid(axis='y')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', edgecolor='0')

    plt.subplot(2,1,2, sharex=ax) # fpage
    x = df_mean_by_target['target_age']
    y = df_mean_by_target['error_fpage']
    std_error_yu4u = df_std_by_target.error_fpage
    plt.scatter(x, y, s=10, zorder=20, label="Age-EG3D")
    plt.plot(x, y)
    plt.fill_between(x, y + std_error_yu4u, y - std_error_yu4u, facecolor='C0', alpha=0.3, label='std')
    # plt.gca().set_box_aspect(1)
    plt.xticks(np.arange(0,110,10))
    plt.title("FPAge model")
    plt.xlabel("Target age")
    plt.ylabel("MAE")
    plt.yticks(np.arange(0,plt.ylim()[1], 5))
    plt.grid(axis='y')
    plt.ylim(0,plt.ylim()[1])
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', edgecolor='0')


    plt.tight_layout()
    fig_name = "pti_comparison"
    plt.savefig(save_path + f"/{fig_name}" + ".png",bbox_inches='tight')
    plt.savefig(save_path + f"/{fig_name}" + ".pgf",bbox_inches='tight')
    print(f"Saved {fig_name} at {save_path}...")