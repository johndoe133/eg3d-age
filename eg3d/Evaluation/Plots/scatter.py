import matplotlib.pyplot as plt
from plot_training_results import plot_setup, compute_figsize
import pandas as pd
import os 
import json
import numpy as np
from training.training_loop import normalize, denormalize

def scatter_plot(network_folder, path):
    plot_setup()

    training_option_path = os.path.join(network_folder, "training_options.json")
    f = open(training_option_path)
    training_option = json.load(f)
    if 'categories' in training_option.keys():
        categories = training_option['categories']
    else: # if evaluating an old version without categories
        categories = [0]
    if 'age_min' in training_option.keys():
        age_min = training_option['age_min']
        age_max = training_option['age_max']
    else:
        age_min = 0
        age_max = 100

    save_path = os.path.join("Evaluation", "Runs", path)
    df = pd.read_csv(os.path.join(save_path, "age_scatter.csv"))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    figsize = compute_figsize(350, 250)
    age_hat = df.age_hat.to_numpy()
    age_hat = denormalize(age_hat, rmin=age_min, rmax=age_max)
    age_true = df.age_true.to_numpy()
    age_true = denormalize(age_true, rmin=age_min, rmax=age_max)
    plt.figure(figsize=figsize, dpi=300)
    plt.scatter(age_true, age_hat, facecolors='none', edgecolors=colors[0])
    plt.plot([age_min, age_max], [age_min, age_max], '--', label="Perfect prediction", color='black')
    plt.xlabel("True age")
    plt.ylabel("Predicted age")
    fig_name = "scatter"
    plt.legend()
    plt.savefig(save_path + f"/{fig_name}" + ".png",bbox_inches='tight')
    plt.savefig(save_path + f"/{fig_name}" + ".pgf",bbox_inches='tight')
    print(f"Figure {fig_name} save at {save_path}")