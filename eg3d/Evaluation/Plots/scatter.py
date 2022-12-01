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
    save_path = os.path.join("Evaluation", "Runs", path)
    df = pd.read_csv(os.path.join(save_path, "age_scatter.csv"))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    age_hat = df.age_hat.to_numpy()
    age_hat_fpage = df.age_hat2.to_numpy()
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
    age_hat = denormalize(age_hat, rmin=age_min, rmax=age_max)
    age_true = denormalize(age_true, rmin=age_min, rmax=age_max)
    idx = normalize(mag, rmin=mag_min, rmax=mag_max, tmin=0)
    colors = viridis(idx)
    # plt.figure(figsize=figsize, dpi=300)
    figsize = compute_figsize(400, 230)
    fig, axs = plt.subplots(1, 2, sharey=True, sharex=False, figsize = figsize, dpi=300, gridspec_kw={'width_ratios': [1, 1.15]})
    # plt.xticks(np.linspace(0, (age_max//10)*10, (age_max//10)+1))
    # plt.yticks(np.linspace(0, (age_max//10)*10, (age_max//10)+1))
    axs[0].scatter(age_true, age_hat, s=5, c=mag, cmap="winter")
    axs[0].set_xlabel("Target age")
    axs[0].set_ylabel("Predicted age")
    axs[0].set_xticks(np.linspace(0, (age_max//10+1)*10, (age_max//10)+2))
    axs[0].set_yticks(np.linspace(0, (age_max//10+1)*10, (age_max//10)+2))
    axs[0].set_title("yu4u model")
    xlim, ylim = axs[0].get_xlim(), axs[0].get_ylim()
    axs[0].plot([-20, 150], [-20, 150], '--', label="Perfect\nprediction", color='black')
    axs[0].set_xlim(xlim); axs[0].set_ylim(ylim)
    axs[0].legend(loc='upper left')
    mapable = axs[1].scatter(age_true, age_hat_fpage, s=5, c=mag, cmap="winter")
    axs[1].set_xlabel("Target age")
    axs[1].set_xticks(np.linspace(0, (age_max//10+1)*10, (age_max//10)+2))
    axs[1].set_title("FPAge model")
    xlim, ylim = axs[1].get_xlim(), axs[1].get_ylim()
    axs[1].plot([-20, 150], [-20, 150], '--', label="Perfect\nprediction", color='black')
    axs[1].set_xlim(xlim); axs[1].set_ylim(ylim)
    axs[1].legend(loc='upper left')

    fig.tight_layout()
    fig.colorbar(mapable, label = "MagFace score")
    fig_name = "scatter"
    fig.savefig(save_path + f"/{fig_name}" + ".png",bbox_inches='tight')
    fig.savefig(save_path + f"/{fig_name}" + ".pgf",bbox_inches='tight')
    print(f"Figure {fig_name} save at {save_path}")

    plt.figure(figsize=compute_figsize(300, 200), dpi=300)
    error = np.abs(age_hat - age_true)
    plt.scatter(error, mag, facecolor='none', edgecolors='steelblue', s=10)
    fig_name = "error_mag"
    plt.xlabel("Absolute age error")
    plt.ylabel("MagFace score")
    plt.savefig(save_path + f"/{fig_name}" + ".png",bbox_inches='tight')
    plt.savefig(save_path + f"/{fig_name}" + ".pgf",bbox_inches='tight')


    save_path_cognitec = os.path.join("Evaluation", "cognitec")
    path2 = path.replace(".",",")
    cognitec_save_path = os.path.join(save_path_cognitec, f"cognitec_{path2}.csv")
    if not os.path.isfile(cognitec_save_path):
        return False #dont make the last plot
    df = pd.read_csv(cognitec_save_path)
    df['target'] = df.meta_data1.apply(lambda x: convert_row(x, "target age"))
    df['magface'] = df.meta_data1.apply(lambda x: convert_row(x, "magface"))
    df['yu4u'] = df.meta_data1.apply(lambda x: convert_row(x, "yu4u"))
    df['prediction'] = df.meta_data2

    x,y, magface_score = df['target'].to_numpy(), df["prediction"].to_numpy(), df['magface'].to_numpy()

    plt.figure(dpi=300, figsize=compute_figsize(330, 230))
    plt.scatter(x,y, s=5, c=magface_score, cmap="winter", zorder=20)
    xlim, ylim = plt.xlim(), plt.ylim()
    plt.xlim(xlim); plt.ylim(ylim)
    plt.plot([-20, 150], [-20, 150], '--', label="Perfect\nprediction", color='black')
    plt.xlabel("Target age")
    plt.ylabel("Predicted age")
    plt.colorbar(label="MagFace score")
    plt.gca().set_aspect(1)
    plt.xticks(np.arange(0,90,10))
    plt.yticks(np.arange(0,90,10))
    plt.legend()
    plt.tight_layout()

    fig_name = "cognitech"
    plt.savefig(save_path + f"/{fig_name}" + ".png",bbox_inches='tight')
    plt.savefig(save_path + f"/{fig_name}" + ".pgf",bbox_inches='tight')
    print(f"Figure {fig_name} save at {save_path}")


def convert_row(row, score):
    if score == "target age":
        s = row.split("-")[0].replace(",",".")
    elif score == "magface":
        s = row.split("-")[-1].replace(",",".").strip(".png")
    elif score == "yu4u":
        s = row.split("-")[1].replace(",",".")
    return float(s)