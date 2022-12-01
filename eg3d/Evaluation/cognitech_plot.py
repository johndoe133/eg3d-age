# from plot_training_results import plot_setup, compute_figsize
import matplotlib.pyplot as plt
import pandas as pd
import os 
import json
import numpy as np
from training.training_loop import normalize, denormalize, get_age_category
import re 
from matplotlib import cm
import seaborn as sn

save_path = os.path.join("Evaluation", "Runs", "cognitech")
df = pd.read_csv(os.path.join(save_path, "cognitec_estimated_ages_no_angles.csv"))

# plot_setup()

df['target'] = df.meta_data1.apply(lambda x: int(x.split(",")[0]))
df['prediction'] = df.meta_data2

x,y = df['target'].to_numpy(), df["prediction"].to_numpy()
plt.figure(dpi=300, figsize=(2,2))
plt.scatter(x,y)
plt.tight_layout()

fig_name = "cognitech"
plt.savefig(save_path + f"/{fig_name}" + ".png",bbox_inches='tight')
plt.savefig(save_path + f"/{fig_name}" + ".pgf",bbox_inches='tight')
print(f"Figure {fig_name} save at {save_path}")