import matplotlib.pyplot as plt
import sys
sys.path.append("/zhome/d7/6/127158/Documents/eg3d-age/eg3d")
from plot_training_results import plot_setup, compute_figsize
import pandas as pd
import os 
import json
import numpy as np
from training.training_loop import normalize, denormalize
from matplotlib import cm
import seaborn as sn
import click 
def add_comma(match):
    return match.group(0) + ','

click.command()
@click.option('--eval_path', help='Eval path name', required=True)
def scatter_plot(
    eval_path: str,
    ):
    plot_setup()

    age_min = 0
    age_max = 75
    eval_path_calibrated = eval_path + "-c"
    df = pd.read_csv(os.path.join(eval_path, "age_scatter.csv"))
    df_cal = pd.read_csv(os.path.join(eval_path_calibrated, "age_scatter.csv"))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    age_hat = df.age_hat.to_numpy()
    age_hat_fpage = df.age_hat2.to_numpy()
    age_true = df.age_true.to_numpy()
    mag = df.mag.to_numpy()
    mag_min, mag_max = mag.min(), mag.max()

    age_hat_cal = df_cal.age_hat.to_numpy()
    age_hat_fpage_cal = df_cal.age_hat2.to_numpy()
    age_true_cal = df_cal.age_true.to_numpy()
    mag_cal = df_cal.mag.to_numpy()
    mag_min_cal, mag_max_cal = mag_cal.min(), mag_cal.max()

    viridis = cm.get_cmap('winter', 12)
    age_hat = denormalize(age_hat, rmin=age_min, rmax=age_max)
    age_true = denormalize(age_true, rmin=age_min, rmax=age_max)
    age_hat_cal = denormalize(age_hat_cal, rmin=age_min, rmax=age_max)
    age_true_cal = denormalize(age_true_cal, rmin=age_min, rmax=age_max)
    
    idx = normalize(mag, rmin=mag_min, rmax=mag_max, tmin=0)
    a,b = np.polyfit(age_true, age_hat,  1)
    a_cal, b_cal = np.polyfit(age_true_cal,age_hat_cal,1)
    colors = viridis(idx)
    # plt.figure(figsize=figsize, dpi=300)
    figsize = compute_figsize(400, 230)
    fig, axs = plt.subplots(1, 2, sharey=True, sharex=False, figsize = figsize, dpi=300, gridspec_kw={'width_ratios': [1, 1.15]})
    
    axs[0].scatter(age_true, age_hat, s=5, c=mag, cmap="winter")
    axs[0].set_xlabel("Target age")
    axs[0].set_ylabel("Predicted age")
    axs[0].set_xticks(np.linspace(0, (age_max//10+1)*10, (age_max//10)+2))
    axs[0].set_yticks(np.linspace(0, (age_max//10+1)*10, (age_max//10)+2))
    axs[0].set_title("Non-calibrated")
    xlim, ylim = axs[0].get_xlim(), axs[0].get_ylim()
    axs[0].plot(np.linspace(-10,110,10), a * np.linspace(-10,110,10) + b, color='r', label="Best fit", zorder=20)
    axs[0].plot([-20, 150], [-20, 150], '--', label="Perfect\nprediction", color='black')
    axs[0].set_xlim(xlim); axs[0].set_ylim(ylim)
    axs[0].legend(loc='upper left')
    
    mapable = axs[1].scatter(age_true_cal, age_hat_cal, s=5, c=mag, cmap="winter")
    axs[1].set_xlabel("Target age")
    axs[1].set_xticks(np.linspace(0, (age_max//10+1)*10, (age_max//10)+2))
    axs[1].set_title("Calibrated")
    xlim, ylim = axs[1].get_xlim(), axs[1].get_ylim()
    axs[1].plot(np.linspace(-10,110,10), a_cal*np.linspace(-10,110,10) + b_cal, color='r', label="Best fit", zorder=20)
    axs[1].plot([-20, 150], [-20, 150], '--', label="Perfect\nprediction", color='black')
    axs[1].set_xlim(xlim); axs[1].set_ylim(ylim)
    axs[1].legend(loc='upper left')

    fig.tight_layout()
    fig.colorbar(mapable, label = "MagFace score")
    fig_name = "scatter_cal"
    fig.savefig(eval_path + f"/{fig_name}" + ".png",bbox_inches='tight')
    fig.savefig(eval_path + f"/{fig_name}" + ".pgf",bbox_inches='tight')
    print(f"Figure {fig_name} save at {eval_path}")


def convert_row(row, score):
    if score == "target age":
        s = row.split("-")[0].replace(",",".")
    elif score == "magface":
        s = row.split("-")[-1].replace(",",".").strip(".png")
    elif score == "yu4u":
        s = row.split("-")[1].replace(",",".")
    return float(s)

if __name__=="__main__":
    scatter_plot('/zhome/d7/6/127158/Documents/eg3d-age/eg3d/Evaluation/Runs/00187-snapshot-001440-trunc-0.75')
