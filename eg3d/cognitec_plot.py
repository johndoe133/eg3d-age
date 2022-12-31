
import matplotlib.pyplot as plt
import pandas as pd
import os 
from plot_training_results import plot_setup, compute_figsize
import numpy as np
from training.training_loop import normalize, denormalize

save_path = os.path.join("Evaluation", "cognitec") #mappen med excel filen
df = pd.read_csv(os.path.join(save_path, "cognitec_estimated_ages_00187-trunc-075.csv"))

plot_setup()

# df['target'] = df.meta_data1.apply(lambda x: int(x.split("_")[-1].strip(".png")))
# df['prediction'] = df.meta_data2
# df['error'] = np.abs(df.prediction - df.target)
# df_mean = df.groupby('target').mean().reset_index()
# x,y = df_mean.target.to_numpy(), df_mean.error.to_numpy()
# plt.figure(dpi=300, figsize=(3,3))
# plt.scatter(x,y, s=5)
# plt.xlabel("Target age")
# plt.ylabel("MAE")
# plt.tight_layout()

# fig_name = "cognitech"
# plt.savefig(save_path + f"/{fig_name}" + ".png",bbox_inches='tight')
# plt.savefig(save_path + f"/{fig_name}" + ".pgf",bbox_inches='tight')
# print(f"Figure {fig_name} save at {save_path}")


df = pd.read_csv(os.path.join(save_path, "cognitec_estimated_ages_scatter_data.txt"))
df.columns=["name", "estimate"]
df['psi'] = df.name.apply(lambda x: float(x.split("-")[2].split("_")[0]))
df['calibrated'] = df.name.apply(lambda x: True if "-c" in x else False)
df['target'] = df.name.apply(lambda x: float(x.split("_")[-1].split("-")[0].replace(",",".")))
df['fancy'] = df.name.apply(lambda x: True if "fancy" in x else False)
df['error'] = np.abs(df.target - df.estimate)
# df['yu4u'] = df.name.apply(lambda x: float(x.split("_")[-1].split("-")[1].replace(",",".")))
df_fancy = df[df.fancy == True]
df_scatter = df[df.fancy != True]
df_scatter['mag'] = df_scatter.name.apply(lambda x: float(x.split("_")[-1].split("-")[-1].strip(".png").replace(",",".")))

figsize = compute_figsize(400, 230)
fig, axs = plt.subplots(1, 2, sharey=True, sharex=False, figsize = figsize, dpi=300, gridspec_kw={'width_ratios': [1, 1.15]})
mask = ((df_scatter.psi == 0.75) & (df_scatter.calibrated==True))
print("Cognitec MAE", df_scatter[mask].error.mean())
a,b = np.polyfit(df_scatter[mask].target, df_scatter[mask].estimate,1)
axs[0].scatter(df_scatter[mask].target, df_scatter[mask].estimate, s=5, c=df_scatter[mask].mag, cmap="winter", alpha=1, zorder=20)
axs[0].set_xlabel("Target age")
axs[0].set_ylabel("Predicted age")
axs[0].set_xticks(np.arange(0,85,10))
axs[0].set_yticks(np.arange(0,85,10))
axs[0].set_title(r"$\texttt{cognitec}$")
xlim, ylim = axs[0].get_xlim(), axs[0].get_ylim()
axs[0].plot(np.linspace(-10,110,10), a*np.linspace(-10,110,10) + b, color='r', label="Best fit", zorder=20)
axs[0].plot([-20, 150], [-20, 150], '--', label="Perfect\nprediction", color='black')
axs[0].set_xlim(xlim); axs[0].set_ylim(ylim)
axs[0].legend(loc='upper left')


fpage_data = r"/zhome/d7/6/127158/Documents/eg3d-age/eg3d/Evaluation/Runs/00187-snapshot-001440-trunc-0.75-c/age_scatter.csv"
df_fpage = pd.read_csv(fpage_data)
age_hat_fpage = df_fpage.age_hat2.to_numpy()
age_true = df_fpage.age_true.to_numpy()
age_true = denormalize(age_true, rmin=0, rmax=75)
mag = df_fpage.mag.to_numpy()
a,b = np.polyfit(age_true, age_hat_fpage, 1)
mapable = axs[1].scatter(age_true, age_hat_fpage, s=5, c=mag, cmap="winter")
axs[1].set_xlabel("Target age")
axs[1].set_xticks(np.arange(0,85,10))
axs[1].set_title(r"$\texttt{FPAge}$")
xlim, ylim = axs[1].get_xlim(), axs[1].get_ylim()
axs[1].plot(np.linspace(-10,110,10), a*np.linspace(-10,110,10) + b, color='r', label="Best fit", zorder=20)
axs[1].plot([-20, 150], [-20, 150], '--', label="Perfect\nprediction", color='black')
axs[1].set_xlim(xlim); axs[1].set_ylim(ylim)
axs[1].legend(loc='upper left')

fig.tight_layout()
fig.colorbar(mapable, label = "MagFace score")
fig_name = "cognitec_scatter"
fig.savefig(save_path + f"/{fig_name}" + ".png",bbox_inches='tight')
fig.savefig(save_path + f"/{fig_name}" + ".pgf",bbox_inches='tight')

print("FPAge MAE:",np.mean(np.abs(age_hat_fpage - age_true)))

truncs = [0.6,0.75,1.0]

fig, axs = plt.subplots(2, 3, sharey=True, sharex=True, figsize = compute_figsize(420,300), dpi=300, gridspec_kw={'width_ratios': [1, 1, 1.15]})
for i, trunc in enumerate(truncs):
    for j, b in enumerate([True, False]):
        axs[j,i].set_xticks(np.arange(0,85,20))
        df_inner = df_scatter[(df_scatter.calibrated==b) & (df_scatter.psi == trunc)]
        x = df_inner.target.to_numpy()
        y = df_inner.estimate.to_numpy()
        mag = df_inner.mag.to_numpy()
        x = x[~np.isnan(y)]
        mag = mag[~np.isnan(y)]
        y = y[~np.isnan(y)]
        mapable = axs[j,i].scatter(x, y, s=5, c=mag, cmap="winter", alpha=1, zorder=1)
        if (i==2 and j==1) or (i==2 and j==0):
            fig.colorbar(mapable, ax=axs[j,i], label = "MagFace score")
        xlim, ylim = axs[j,i].get_xlim(), axs[j,i].get_ylim()
        a,b = np.polyfit(x, y, 1)
        axs[j,i].plot([-20, 150], [-20, 150], '--', label="Perfect\nprediction", color='black', zorder=20)
        axs[j,i].plot(np.linspace(-10,110,10), a*np.linspace(-10,110,10) + b, color='r', label="Best fit", zorder=20)
        axs[j,i].set_xlim(xlim)
        axs[j,i].set_ylim(ylim)
    
    if i==0:
        axs[0,i].set_ylabel("Predicted age")
        axs[1,i].set_ylabel("Predicted age")
    if j==1:
        axs[1,i].set_xlabel("Target age")
    pad=5
    for ax, angle in zip(axs[0], truncs):
        text = f"$\psi$={angle}"
        ax.annotate(text, xy=(0.5, 1), xytext=(0, pad), 
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

    for ax, angle in zip(axs[:,0], ["Calibrated", "Not-calibrated"]):
        text = f"{angle}"
        ax.annotate(text, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0), rotation=90,
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')

fig_name = "cognitec_many"
fig.tight_layout()
fig.savefig(save_path + f"/{fig_name}" + ".png",bbox_inches='tight')
fig.savefig(save_path + f"/{fig_name}" + ".pgf",bbox_inches='tight')
