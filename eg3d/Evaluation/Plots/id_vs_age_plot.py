import matplotlib.pyplot as plt
import pandas as pd
import os 
import json
import numpy as np
from plot_training_results import plot_setup, compute_figsize
from training.training_loop import normalize, denormalize, get_age_category



def plot_id_vs_age_scatter(network_folder, path):
    save_path = os.path.join("Evaluation", "Runs", path)
    df = pd.read_csv(os.path.join(save_path, "age_scatter.csv"))

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

    age_diff = df.age_diff.to_numpy()
    cos_sim = df.cos_sim.to_numpy()

    plt.figure(dpi=300)
    plot_setup()
    plt.scatter(age_diff, cos_sim, s=2)
    plt.xlabel('Absolute age difference')
    plt.ylabel('Cosine similarity')
    
    fig_name = 'age_cos_scatter'

    plt.savefig(save_path + f"/{fig_name}" + ".png",bbox_inches='tight')
    plt.savefig(save_path + f"/{fig_name}" + ".pgf",bbox_inches='tight')
    print(f"Figure {fig_name} save at {save_path}")
