import matplotlib.pyplot as plt
from plot_training_results import plot_setup, compute_figsize
import pandas as pd
import os 
from sklearn.neighbors import KernelDensity
import numpy as np
from training.training_loop import normalize, denormalize
import json

def truncation_plot(network_folder, save_path):
    plot_setup()

    training_option_path = os.path.join(network_folder, "training_options.json")
    f = open(training_option_path)
    training_option = json.load(f)

    if 'age_min' in training_option.keys():
        age_min = training_option['age_min']
        age_max = training_option['age_max']
    else:
        age_min = 0
        age_max = 100

    save_path = os.path.join("Evaluation", "Runs", save_path)
    df = pd.read_csv(os.path.join(save_path, "truncation_data.csv"))

    df.age_hat = df.age_hat.apply(lambda x: denormalize(x, rmin=age_min, rmax=age_max))
    df['error'] = np.abs(df.age - df.age_hat)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    grouped_data = df.groupby('truncation_psi')
    error = grouped_data.mean().error.to_numpy()
    error_std = grouped_data.std().error.to_numpy()
    mean_mag = grouped_data.mean().mag.to_numpy()

    truncation = df.truncation_psi.unique()

    figsize = compute_figsize(350, 250)
    plt.figure(dpi=300, figsize=figsize)

    plt.scatter(truncation, error, s=10, c=mean_mag, cmap="winter", zorder=20)
    plt.plot(truncation, error, color="black", alpha=0.6, zorder=1)
    plt.fill_between(truncation, error - error_std, error + error_std, alpha=0.4, label="std")
    plt.xlabel("Truncation factor")
    plt.ylabel(r"Mean absolute age error")
    plt.colorbar(label = "MagFace magnitude")
    plt.legend()
    fig_name = "truncation"
    plt.savefig(save_path + f"/{fig_name}" + ".png",bbox_inches='tight')
    plt.savefig(save_path + f"/{fig_name}" + ".pgf",bbox_inches='tight')
    print(f"Figure {fig_name} save at {save_path}")

    plt.figure(dpi=300, figsize=figsize)
    plt.scatter(truncation, mean_mag)
    plt.xlabel("Truncation factor")
    plt.ylabel(r"MagFace magnitude")
    fig_name = "magvstrunc"
    plt.savefig(save_path + f"/{fig_name}" + ".png",bbox_inches='tight')
    plt.savefig(save_path + f"/{fig_name}" + ".pgf",bbox_inches='tight')
    print(f"Figure {fig_name} save at {save_path}")

