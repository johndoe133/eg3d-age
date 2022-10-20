import matplotlib.pyplot as plt
from plot_training_results import plot_setup, compute_figsize
import pandas as pd
import os 
import json
import numpy as np
from training.training_loop import normalize, denormalize, get_age_category
import re 
from matplotlib import cm
import seaborn as sn

def add_comma(match):
    return match.group(0) + ','

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
    age_loss_fn = training_option['age_loss_fn']
    fig_name = "scatter"
    save_path = os.path.join("Evaluation", "Runs", path)
    df = pd.read_csv(os.path.join(save_path, "age_scatter.csv"))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    age_hat = df.age_hat.to_numpy()
    age_true = df.age_true.to_numpy()
    mag = df.mag.to_numpy()
    mag_min, mag_max = mag.min(), mag.max()

    # if age_loss_fn == "CAT":
    # ranges = 101 #todo
    # confusion = np.zeros((ranges, ranges))
    # for i, age in enumerate(age_hat):
    #     predicted_range = get_age_category(np.array([age]), categories, normalize_category=True)
    #     s = age_true[i]
    #     s = re.sub(r'\[[0-9\.\s]+\]', add_comma, s)
    #     s = re.sub(r'([0-9\.]+)', add_comma, s)
    #     true_range = eval(s)[0]
    #     x = np.array(predicted_range).nonzero()[0][0]
    #     y = np.array(true_range).nonzero()[0][0]
    #     confusion[x,y] = confusion[x,y] + 1
    # figsize = compute_figsize(350, 250)
    # plt.figure(figsize=figsize, dpi=300)
    # labels = []
    # for i in range(ranges):
    #     labels.append(f"{categories[i]}-{categories[i+1] - 1}")
    # df_cm = pd.DataFrame(confusion, index = labels, columns = labels)

    # sn.set(font_scale=1.4) # for label size
    # sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

    # else:
    viridis = cm.get_cmap('winter', 12)
    figsize = compute_figsize(350, 250)
    age_hat = denormalize(age_hat, rmin=age_min, rmax=age_max)
    age_true = denormalize(age_true, rmin=age_min, rmax=age_max)
    idx = normalize(mag, rmin=mag_min, rmax=mag_max, tmin=0)
    colors = viridis(idx)
    corr, _ = pearsonr(age_hat, age_true)
    corr = round(corr, 4)
    plt.figure(figsize=figsize, dpi=300)
    plt.xticks(np.linspace(0, (age_max//10)*10, (age_max//10)+1))
    plt.yticks(np.linspace(0, (age_max//10)*10, (age_max//10)+1))
    plt.scatter(age_true, age_hat, s=5, c=mag, cmap="winter")
    plt.plot([age_min, age_max], [age_min, age_max], '--', label="Perfect prediction", color='black')
    plt.xlabel("True age")
    plt.ylabel("Predicted age")
    plt.colorbar(label = "MagFace magnitude")
    plt.legend()
    plt.savefig(save_path + f"/{fig_name+str(corr)}" + ".png",bbox_inches='tight')
    plt.savefig(save_path + f"/{fig_name+str(corr)}" + ".pgf",bbox_inches='tight')
    print(f"Figure {fig_name+str(corr)} save at {save_path}")
