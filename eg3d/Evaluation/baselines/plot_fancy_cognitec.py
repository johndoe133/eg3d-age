import matplotlib.pyplot as plt
import pandas as pd
import os 
import sys
home = str(os.path.expanduser('~'))
sys.path.append(os.path.join(home, 'Documents/eg3d-age/eg3d/'))

from plot_training_results import plot_setup
import numpy as np


# age-eg3d optimal model, 20-30
save_path = os.path.join("Evaluation", "cognitec") #mappen med excel filen
df = pd.read_csv("/zhome/d1/9/127646/Documents/eg3d-age/eg3d/Evaluation/baselines/cognitec_estimated_ages_00187-trunc-075.csv")

list_20_30 = os.listdir("/work3/s174379/datasets/20_30")
list_50_60 = os.listdir("/work3/s174379/datasets/50_60")

plot_setup()

df['target'] = df.meta_data1.apply(lambda x: int(x.split("_")[-1].strip(".png")))
df["20_30"] = df.meta_data1.apply(lambda x: bool((x.split("_")[1] + ".jpg") in list_20_30))
df['prediction'] = df.meta_data2
df['error'] = np.abs(df.prediction - df.target)

# find MAE 
age_eg3d_mae = np.mean(df['error'])
age_eg3d_mae_20_30 = np.mean(df[df["20_30"] == True].error)
age_eg3d_mae_50_60 = np.mean(df[df["20_30"] == False].error)

df_mean_20_30 = df[df["20_30"] == True].groupby('target').mean().reset_index()
x,y = df_mean_20_30.target.to_numpy(), df_mean_20_30.error.to_numpy()
plt.figure(dpi=300, figsize=(5,3))
plt.scatter(x,y, s=5, c="C0", label="Age-EG3D")
plt.plot(x, y, c="C0")

baselines = pd.read_csv("/zhome/d1/9/127646/Documents/eg3d-age/eg3d/Evaluation/baselines/baselines_cognitec.csv")
baselines["target"] = baselines.meta_data1.apply(lambda x: int(x.split("_")[3]))
baselines["20_30"] = baselines.meta_data1.apply(lambda x: "20_30" in x)
baselines["prediction"] = baselines.meta_data2
baselines["error"] = np.abs(baselines["target"] - baselines["prediction"])
baselines["model_name"] = baselines.meta_data1.apply(lambda x: x.split("_")[0] + "_" + x.split("_")[1] + "_" + x.split("_")[2])

# find MAEs
sam_mae = np.mean(baselines[(baselines["model_name"] == "20_30_sam") | (baselines["model_name"] == "50_60_sam")].error)
cusp_mae = np.mean(baselines[(baselines["model_name"] == "20_30_cusp") | (baselines["model_name"] == "50_60_cusp")].error)
hrfae_mae = np.mean(baselines[(baselines["model_name"] == "20_30_hrfae") | (baselines["model_name"] == "50_60_hrfae")].error)

sam_mae_20_30 = np.mean(baselines[baselines["model_name"] == "20_30_sam"].error)
cusp_mae_20_30 = np.mean(baselines[baselines["model_name"] == "20_30_cusp"].error)
hrfae_mae_20_30 = np.mean(baselines[baselines["model_name"] == "20_30_hrfae"].error)

sam_mae_50_60 = np.mean(baselines[baselines["model_name"] == "50_60_sam"].error)
cusp_mae_50_60 = np.mean(baselines[baselines["model_name"] == "50_60_cusp"].error)
hrfae_mae_50_60 = np.mean(baselines[baselines["model_name"] == "50_60_hrfae"].error)

# print MAEs
print("general MAE & 20-30 MAE & 50-60 MAE")
print(f"age-EG3D &  {age_eg3d_mae} & {age_eg3d_mae_20_30} & {age_eg3d_mae_50_60}")
print(f"SAM &  {sam_mae} & {sam_mae_20_30} & {sam_mae_50_60}")
print(f"HRFAE &  {hrfae_mae} & {hrfae_mae_20_30} & {hrfae_mae_50_60}")
print(f"CUSP &  {cusp_mae} & {cusp_mae_20_30} & {cusp_mae_50_60}\n")

# print("20-30 MAE")
# print(f"age_eg3d: {age_eg3d_mae_20_30}")
# print(f"sam: {sam_mae_20_30}")
# print(f"hrfae: {hrfae_mae_20_30}")
# print(f"cusp: {cusp_mae_20_30}\n")

# print("50-60 MAE")
# print(f"age_eg3d: {age_eg3d_mae_50_60}")
# print(f"sam: {sam_mae_50_60}")
# print(f"hrfae: {hrfae_mae_50_60}")
# print(f"cusp: {cusp_mae_50_60}")

i = 0
for model_name in sorted(baselines["model_name"].unique()):
    if "20_30" in model_name:
        i += 1
        baselines_model = baselines[baselines["model_name"] == model_name].groupby("target").mean().reset_index()
        x,y = baselines_model.target.to_numpy(), baselines_model.error.to_numpy()
        plt.scatter(x,y, s=5, c=f"C{i}", label=model_name.split("_")[-1].upper())
        plt.plot(x, y, c=f"C{i}")

plt.legend()
plt.xticks(np.arange(0,105,5))
plt.ylim(0, plt.ylim()[1])
plt.yticks(np.arange(0, plt.ylim()[1],5))
plt.xlabel("Target age")
plt.ylabel("MAE")
plt.tight_layout()
plt.savefig("/zhome/d1/9/127646/Documents/eg3d-age/eg3d/Evaluation/baselines/fancy_20_30.png")



# 50-60 age-eg3d optimal model

df_mean_50_60 = df[df["20_30"] == False].groupby('target').mean().reset_index()
x,y = df_mean_50_60.target.to_numpy(), df_mean_50_60.error.to_numpy()
plt.figure(dpi=300, figsize=(5,3))
plt.scatter(x,y, s=5, c="C0", label="age-EG3D")
plt.plot(x, y, c="C0")

i = 0
for model_name in sorted(baselines["model_name"].unique()):
    if "50_60" in model_name:
        i += 1
        baselines_model = baselines[baselines["model_name"] == model_name].groupby("target").mean().reset_index()
        x,y = baselines_model.target.to_numpy(), baselines_model.error.to_numpy()
        plt.scatter(x,y, s=5, c=f"C{i}", label=model_name.split("_")[-1].upper())
        plt.plot(x, y, c=f"C{i}")

plt.legend()
plt.xticks(np.arange(0,105,5))
plt.ylim(0, plt.ylim()[1])
plt.yticks(np.arange(0, plt.ylim()[1],5))
plt.xlabel("Target age")
plt.ylabel("MAE")
plt.tight_layout()
plt.savefig("/zhome/d1/9/127646/Documents/eg3d-age/eg3d/Evaluation/baselines/fancy_50_60.png")


# plot fpage 20-30
df = pd.read_csv("/zhome/d1/9/127646/Documents/eg3d-age/eg3d/Evaluation/baselines/baselines.csv")

list_20_30 = os.listdir("/work3/s174379/datasets/20_30")
list_50_60 = os.listdir("/work3/s174379/datasets/50_60")

df["is_20_30"] = True
df.loc[df["age_model"] == "age_eg3d", "is_20_30"] = df.loc[df["age_model"] == "age_eg3d", "image_name"].apply(lambda x:  bool((x.split("_")[1] + ".jpg") in list_20_30))
df.loc[df["age_model"] != "age_eg3d", "is_20_30"] = df.loc[df["age_model"] != "age_eg3d", "age_model"].apply(lambda x:  "20_30" in x)

df["error"] = np.abs(df["target_age"] - df["age_hat_fpage"])


d = {"age_eg3d": "Age-EG3D", 
        "sam_20_30": "SAM", "sam_50_60": "SAM",
        "hrfae_20_30": "HRFAE", "hrfae_50_60": "HRFAE",
        "cusp_20_30": "CUSP", "cusp_50_60": "CUSP"
    }

plt.figure(dpi=300, figsize=(5,3))
plot_setup()

df_20_30 = df[df["is_20_30"] == True]

for i, model_name in enumerate(sorted(df_20_30.age_model.unique())):
    df_mean_20_30 = df_20_30[(df_20_30["age_model"] == model_name)].groupby('target_age').mean().reset_index()
    x,y = df_mean_20_30.target_age.to_numpy(), df_mean_20_30.error.to_numpy()
    
    plt.scatter(x,y, s=5, c=f"C{i}", label=f"{d[model_name]}")
    plt.plot(x, y, c=f"C{i}")

plt.legend()
plt.xticks(np.arange(0,105,5))
plt.ylim(0, plt.ylim()[1])
plt.yticks(np.arange(0, plt.ylim()[1],5))
plt.xlabel("Target age")
plt.ylabel("MAE")
plt.tight_layout()
plt.savefig("/zhome/d1/9/127646/Documents/eg3d-age/eg3d/Evaluation/baselines/fancy_20_30_fpage.png")


# fpage 50-60
plt.figure(dpi=300, figsize=(5,3))
plot_setup()

df_50_60 = df[df["is_20_30"] == False]

for i, model_name in enumerate(sorted(df_50_60.age_model.unique())):
    df_mean_20_30 = df_50_60[(df_50_60["age_model"] == model_name)].groupby('target_age').mean().reset_index()
    x,y = df_mean_20_30.target_age.to_numpy(), df_mean_20_30.error.to_numpy()
    
    plt.scatter(x,y, s=5, c=f"C{i}", label=f"{d[model_name]}")
    plt.plot(x, y, c=f"C{i}")

plt.legend()
plt.xticks(np.arange(0,105,5))
plt.ylim(0, plt.ylim()[1])
plt.yticks(np.arange(0, plt.ylim()[1],5))
plt.xlabel("Target age")
plt.ylabel("MAE")
plt.tight_layout()
plt.savefig("/zhome/d1/9/127646/Documents/eg3d-age/eg3d/Evaluation/baselines/fancy_50_60_fpage.png")



# cos sim 20-30
plt.figure(dpi=300, figsize=(5,3))
plot_setup()

df_20_30 = df[df["is_20_30"] == True]

for i, model_name in enumerate(sorted(df_20_30.age_model.unique())):
    df_mean_20_30 = df_20_30[(df_20_30["age_model"] == model_name)].groupby('target_age').mean().reset_index()
    x,y = df_mean_20_30.target_age.to_numpy(), df_mean_20_30.cos_sim.to_numpy()
    
    plt.scatter(x,y, s=5, c=f"C{i}", label=f"{d[model_name]}")
    plt.plot(x, y, c=f"C{i}")

plt.legend()
plt.xticks(np.arange(0,105,5))
plt.ylim(0, plt.ylim()[1])
plt.yticks(np.arange(0, plt.ylim()[1],0.2))
plt.xlabel("Target age")
plt.ylabel("Cosine similarity")
plt.tight_layout()
plt.savefig("/zhome/d1/9/127646/Documents/eg3d-age/eg3d/Evaluation/baselines/fancy_20_30_cos_sim.png")


# cos sim 50-60
plt.figure(dpi=300, figsize=(5,3))
plot_setup()

df_50_60 = df[df["is_20_30"] == False]

for i, model_name in enumerate(sorted(df_50_60.age_model.unique())):
    df_mean_20_30 = df_50_60[(df_50_60["age_model"] == model_name)].groupby('target_age').mean().reset_index()
    x,y = df_mean_20_30.target_age.to_numpy(), df_mean_20_30.cos_sim.to_numpy()
    
    plt.scatter(x,y, s=5, c=f"C{i}", label=f"{d[model_name]}")
    plt.plot(x, y, c=f"C{i}")

plt.legend()
plt.xticks(np.arange(0,105,5))
plt.ylim(0, plt.ylim()[1])
plt.yticks(np.arange(0, plt.ylim()[1],0.2))
plt.xlabel("Target age")
plt.ylabel("Cosine similarity")
plt.tight_layout()
plt.savefig("/zhome/d1/9/127646/Documents/eg3d-age/eg3d/Evaluation/baselines/fancy_50_60_cos_sim.png")


