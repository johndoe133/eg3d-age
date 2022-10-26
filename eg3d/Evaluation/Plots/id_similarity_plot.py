import matplotlib.pyplot as plt
from plot_training_results import plot_setup, compute_figsize
import pandas as pd
import os 
from sklearn.neighbors import KernelDensity
import numpy as np

def id_plot(save_path):
    plot_setup()
    save_path = os.path.join("Evaluation", "Runs", save_path)
    df = pd.read_csv(os.path.join(save_path, "id_evaluation.csv"))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    figsize = compute_figsize(426, 500)
    #id plot
    bandwidth=.005
    xlim = df.cosine_sim.min()
    xlim = xlim - (xlim*0.05)
    pad = 5 # in points
    ages = df.age1.unique()
    fig, axs = plt.subplots(len(ages), len(ages), sharex=True, sharey=True, figsize = figsize, dpi=300)
    for i, age1 in enumerate(ages):
        df_age = df[df.age1==age1]
        for j, age2 in enumerate(ages):
            ## AXIS LABELS
            if axs[i,j].get_subplotspec().is_last_row():
                axs[i,j].set_xlabel("Cosine\nsimilarity")

            if axs[i,j].get_subplotspec().is_first_row():
                text = f"Age: {int(age2)}"
                axs[i,j].annotate(text, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

            if axs[i,j].get_subplotspec().is_first_col():
                axs[i,j].set_ylabel("Density")
                text = f"Age:\n{int(age1)}"
                axs[i,j].annotate(text, xy=(0, 0.5), xytext=(-axs[i,j].yaxis.labelpad - pad, 0),
                    xycoords=axs[i,j].yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')
            
            if age1==age2:
                # skip comparing same image
                continue

            cos = df_age[df_age.age2 == age2].cosine_sim.to_numpy()
            kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
            kde.fit(cos[:, None])
            x_d = np.linspace(xlim, 1, 200)
            logprob = kde.score_samples(x_d[:, None])
            pdf = np.exp(logprob)
            axs[i,j].plot(x_d, pdf)
            axs[i,j].fill_between(x_d, pdf, alpha=0.5)
            axs[i,j].set_xlim(xlim, 1)
            
    fig.tight_layout()
    plt.subplots_adjust(
                    wspace=0.1,
                    hspace=0.1)
                    
    fig_name = "id.png"
    plt.savefig(save_path + f"/{fig_name}")
    print(f"Figure {fig_name} save at {save_path}")  