import matplotlib.pyplot as plt
from plot_training_results import plot_setup, compute_figsize
import pandas as pd
import os 
from sklearn.neighbors import KernelDensity
import numpy as np

def set_age_plot(save_path):
    plot_setup()
    save_path = os.path.join("Evaluation", "Runs", save_path)
    df = pd.read_csv(os.path.join(save_path, "age_evaluation.csv"))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    figsize = compute_figsize(426, 500)
    fig, axs = plt.subplots(3, 3, sharex=True, sharey=True, figsize = figsize, dpi=300)
    ages = df.age.unique()
    for i, age in enumerate(ages):
        df_age = df[df.age==age]
        age_hat = df_age.age_hat.to_numpy()
        true_age = df_age.age.to_numpy()
        age_hat = age_hat[:, np.newaxis]

        #Density plot
        kde = KernelDensity(kernel="gaussian", bandwidth=0.5).fit(age_hat)
        x = np.linspace(-5,90, 1000)[:, np.newaxis]
        log_dens = kde.score_samples(x)
        axs.ravel()[i].fill_between(x[:, 0], np.zeros(1000), np.exp(log_dens), color=colors[i], edgecolor=None, alpha=0.8)

        ylimmin, ylimmax = axs.ravel()[i].get_ylim()
        axs.ravel()[i].vlines(int(age), 0, 1, linestyles = '--', colors="black", label=f"Set age")
        axs.ravel()[i].set_ylim(ylimmin, 0.1)

        axs.ravel()[i].set_xticks(np.linspace(0, 80, 5))
         # Axis labels
        if axs.ravel()[i].get_subplotspec().is_first_col():
            axs.ravel()[i].set_ylabel("Normalized Density")
        if axs.ravel()[i].get_subplotspec().is_last_row():
            axs.ravel()[i].set_xlabel("Estimated age")
        axs.ravel()[i].legend()

    fig.tight_layout()
    fig_name = "grid.png"
    plt.savefig(save_path + f"/{fig_name}")
    print(f"Figure {fig_name} save at {save_path}")