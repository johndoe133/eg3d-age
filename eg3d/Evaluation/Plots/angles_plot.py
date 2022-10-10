import matplotlib.pyplot as plt
from plot_training_results import plot_setup, compute_figsize
import pandas as pd
import os 
from sklearn.neighbors import KernelDensity
import numpy as np

def angles_plot(save_path, angles_p, angles_y):
    plot_setup()
    save_path = os.path.join("Evaluation", "Runs", save_path)
    figsize = compute_figsize(426, 500)
    df = pd.read_csv(os.path.join(save_path, "age_evaluation.csv"))
    fig, axs = plt.subplots(3, 3, sharex=True, sharey=True, figsize = figsize, dpi=300)

    pad = 5 # in points

    for ax, angle in zip(axs[0], angles_y):
        text = f"$Angle_y$={angle}"
        ax.annotate(text, xy=(0.5, 1), xytext=(0, pad), 
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

    for ax, angle in zip(axs[:,0], angles_p):
        text = f"$Angle_p$={angle}"
        ax.annotate(text, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0), rotation=90,
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')
        
    for i, angle_y in enumerate(angles_y):
        for j, angle_p in enumerate(angles_p):
            df_angle = df[(df.angle_y==angle_y) & (df.angle_p==angle_p)]
            error = df_angle.error.to_numpy()
            # axs[i,j].hist(error, np.linspace(error.min(), error.max(), 20))
            error = error[:, np.newaxis]
            kde = KernelDensity(kernel="gaussian", bandwidth=1).fit(error)
            x = np.linspace(error.min() * 1.1, error.max() * 1.1, 1000)[:, np.newaxis]
            log_dens = kde.score_samples(x)
            axs[i,j].fill(x[:, 0], np.exp(log_dens), alpha=0.8)
            # axs[i,j].fill_between(x[:,0], np.zeros(1000), np.exp(log_dens))
            
            # Axis labels
            if axs[i,j].get_subplotspec().is_first_col():
                axs[i,j].set_ylabel("Normalized Density")
            if axs[i,j].get_subplotspec().is_last_row():
                axs[i,j].set_xlabel("Error")
            # axs[i,j].legend()
    fig.tight_layout()
    fig_name = "angles.png"
    plt.savefig(save_path + f"/{fig_name}")
    print(f"Figure {fig_name} save at {save_path}")