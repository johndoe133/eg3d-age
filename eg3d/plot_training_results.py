import os
import click
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd    
import time
import json

@click.command()
@click.option('--training_run', type=str, help='Past in folder name in training run', required=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=False)
@click.option('--figsize', help='Figsize of the image', type=tuple, default=(10,7))
@click.option('--dpi', help='Dots per image for image', type=int, default=300)
@click.option('--name', help='Name of the output image', type=str, default="training_results.png")
@click.option('--id', help='If id loss is available', type=bool, default=True)
def generate_images(
    training_run: str,
    outdir: str,
    figsize: tuple,
    dpi: int,
    name: str,
    id: bool,
):
    if not(outdir):
        outdir = training_run # save in same folder as training
    
    print(f'Creating plots for training run folder:"{training_run}"...')
    #Load data 
    root = './training-runs'
    json_filename = 'stats.jsonl'
    path = os.path.join(root, training_run, json_filename)
    stats = pd.read_json(path, lines=True)
    loss_scores_fake, loss_G, loss_D, r1_penalty, loss_age, loss_id, hours = [], [], [], [], [], [], []
    loss_G_std, loss_D_std, loss_age_std, loss_id_std = [],[],[],[]
    for row in stats.iterrows():
        data = row[1]
        loss_scores_fake.append(data['Loss/scores/fake']['mean'])
        loss_G.append(data['Loss/G/loss']['mean'])
        loss_G_std.append(data['Loss/G/loss']['std'])
        loss_D.append(data['Loss/D/loss']['mean'])
        loss_D_std.append(data['Loss/D/loss']['std'])
        r1_penalty.append(data['Loss/r1_penalty']['mean'])
        loss_age.append(data['Loss/scores/age']['mean'])
        loss_age_std.append(data['Loss/scores/age']['std'])
        if id:
            loss_id.append(data['Loss/scores/id']['mean'])
            loss_id_std.append(data['Loss/scores/id']['std'])
        else:
            loss_id.append(0)
            loss_id_std.append(0)
        hours.append(data['Timing/total_hours']['mean'])
    
    fid = []
    fid_filename = 'metric-fid50k_full.jsonl'
    path_fid = os.path.join(root, training_run, fid_filename)
    fid_df = pd.read_json(path_fid, lines=True)
    for row in fid_df.iterrows():
        fid.append(row[1]["results"]["fid50k_full"])
    
    training_option_path = os.path.join(root, training_run, "training_options.json")
    f = open(training_option_path)
    training_option = json.load(f)
    id_scale = training_option['id_scale']
    age_scale = training_option['age_scale']
    f.close()

    #convert to numpy
    loss_scores_fake, loss_G, loss_D, r1_penalty, loss_age, loss_id, hours = np.array(loss_scores_fake), np.array(loss_G), np.array(loss_D), np.array(r1_penalty), np.array(loss_age), np.array(loss_id), np.array(hours)
    loss_G_std, loss_D_std, loss_age_std, loss_id_std = np.array(loss_G_std), np.array(loss_D_std), np.array(loss_age_std), np.array(loss_id_std)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.tight_layout()
    x = range(len(hours))
    
    #GD_ylim = max(list(loss_G + loss_G_std) + list(loss_D + loss_D_std)) * 1.10
    ## AGE LOSS ##
    axs1 = fig.add_subplot(3,2,1)
    axs1.plot(x, loss_age, label="Loss")
    age_loss_ylim = max(loss_age + loss_age_std)
    axs1.set_ylabel("Age loss")
    axs1.set_ylim(0, age_loss_ylim*1.05)
    axs1.fill_between(x, loss_age - loss_age_std, loss_age + loss_age_std, alpha=0.4, label="std")
    axs1.set_title(f"With age_scale = {age_scale}")
    axs1.legend()

    ## ID LOSS ##
    axs2 = fig.add_subplot(3,2,2)
    axs2.plot(x, loss_id, label="Loss")
    if id:
        id_loss_ylim = max(loss_id + loss_id_std)
    else:
        id_loss_ylim = 1
    axs2.set_ylabel("ID loss")
    axs2.set_ylim(0, id_loss_ylim*1.05)
    axs2.fill_between(x, loss_id - loss_id_std, loss_id + loss_id_std, alpha=0.4, label="std")
    axs2.set_title(f"With id_scale = {id_scale}")
    axs2.legend()

    ## GENERATOR LOSS ##
    axs3 = fig.add_subplot(3,2,3)
    axs3.set_ylabel("G loss")
    axs3.fill_between(x, loss_G - loss_G_std, loss_G + loss_G_std, alpha=0.4, label="std")
    axs3.set_ylim(0,max(loss_G + loss_G_std)*1.10)
    axs3.plot(x, loss_G, label="Loss")
    axs3.legend()    

    ## DISCRIMINATOR LOSS ##
    axs4 = fig.add_subplot(3,2,4)
    axs4.plot(x, loss_D, label="Loss")
    axs4.set_ylabel("D loss")
    axs4.set_ylim(0,max(loss_D + loss_D_std)*1.10)
    axs4.fill_between(x, loss_D - loss_D_std, loss_D + loss_D_std, alpha=0.4, label="std")
    axs4.legend()

    ## FID score
    axs5 = fig.add_subplot(3,2,5)
    axs5.set_ylabel("FID Score")
    axs5.bar(range(len(fid)), fid, zorder=20)
    axs5.set_xticks([])
    axs5.grid(axis='y')

    
    # save figure
    plot_name = name
    plot_path = os.path.join(root, outdir, plot_name)
    fig.tight_layout()
    fig.savefig(plot_path)

    print("Ending program...")

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter