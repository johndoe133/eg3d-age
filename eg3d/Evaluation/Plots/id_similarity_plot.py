import matplotlib.pyplot as plt
from plot_training_results import plot_setup, compute_figsize
import pandas as pd
import os 
from sklearn.neighbors import KernelDensity
import numpy as np
from sklearn.preprocessing import StandardScaler

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

    # second id plot
    # scaling
    scaler = StandardScaler()
    cosine_sim = df.cosine_sim.to_numpy()
    cosine_sim_arcface = df.cosine_sim_arcface.to_numpy()

    scaler.fit(cosine_sim.reshape(-1,1))
    cosine_sim_transformed = scaler.transform(cosine_sim.reshape(-1,1))
    df['cosine_sim_transformed'] = cosine_sim_transformed.reshape(-1)

    scaler.fit(cosine_sim_arcface.reshape(-1,1))
    cosine_sim_arcface_transformed = scaler.transform(cosine_sim_arcface.reshape(-1,1))
    df['cosine_sim_arcface_transformed'] = cosine_sim_arcface_transformed.reshape(-1)

    training_id_model = df.id_train.iloc[0]
    figsize = compute_figsize(340, 550)
    fig, axs = plt.subplots(len(ages), 1, sharey=False, sharex=True, figsize = figsize, dpi=300)
    df_grouped = df.groupby(["age1","age2"]).mean().reset_index()
    df_grouped_std = df.groupby(["age1","age2"]).std().reset_index()
    for i, age1 in enumerate(ages):
        cosine_sim = df_grouped[df_grouped.age1 == age1].cosine_sim.to_list()
        std = df_grouped_std[df_grouped_std.age1 == age1].cosine_sim.to_list()
        x = df_grouped[df_grouped.age1 == age1].age2.to_list()

        # x.insert(i, age1)
        cosine_sim_arcface = df_grouped[df_grouped.age1 == age1].cosine_sim_arcface.to_list()
        std_arcface = df_grouped_std[df_grouped_std.age1 == age1].cosine_sim_arcface.to_list()
        
        # cosine_sim.insert(i,1)
        # std.insert(i,0)
        cosine_sim[i]=1
        std[i] = 0
        # cosine_sim_arcface.insert(i,1)
        # std_arcface.insert(i,0)
        cosine_sim_arcface[i]=1
        std_arcface[i]=0

        x, cosine_sim, std, cosine_sim_arcface, std_arcface = np.array(x), np.array(cosine_sim), np.array(std), np.array(cosine_sim_arcface), np.array(std_arcface)

        axs[i].scatter(x, cosine_sim, s=15, zorder=20, label=training_id_model)
        axs[i].plot(x, cosine_sim, color="black", alpha=0.6, zorder=1)
        axs[i].fill_between(x, cosine_sim - std, cosine_sim + std, alpha=0.4, label="std", facecolor="C0")
        
        axs[i].scatter(x, cosine_sim_arcface, s=15, zorder=15, label="ArcFace")
        axs[i].plot(x, cosine_sim_arcface, color="black", alpha=0.6, zorder=1)
        axs[i].fill_between(x, cosine_sim_arcface - std_arcface, cosine_sim_arcface + std_arcface, alpha=0.4, label="std", facecolor="C1")
        # axs[i].set_ylim(0,1.1)
        axs[i].set_ylabel(r"Average $S_C$")
        if axs[i].get_subplotspec().is_last_row():
            axs[i].set_xlabel("Age")
        axs[i].set_xticks(ages) 
        axs[i].set_xticklabels(ages)
        axs[i].set_title(f"Considered age: {ages[i]}")

        xlim = axs[i].get_xlim()

        axs[i].hlines(0.8729, -10, 120, colors='C0', linestyles = 'dashed', label="MagFace\nNon-mated\naverage $S_C$")
        axs[i].hlines(0.0107, -10, 120, colors='C1', linestyles = 'dashed', label="ArcFace\nNon-mated\naverage $S_C$")
        axs[i].set_xlim(xlim)
        if i==2:
            legend = axs[i].legend(bbox_to_anchor=(1.36, 0.5), loc='center right', edgecolor='0')

    # fig.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)
    fig_name = "id2.png"
    plt.savefig(save_path + f"/{fig_name}", bbox_inches="tight")
    print(f"Figure {fig_name} save at {save_path}")  