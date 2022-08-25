import os
import click
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd    
import time

@click.command()
@click.option('--training_run', type=str, help='Past in folder name in training run', required=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=False)
@click.option('--figsize', help='Figsize of the image', type=tuple, default=(10,7))
@click.option('--dpi', help='Dots per image for image', type=int, default=300)
@click.option('--name', help='Name of the output image', type=str, default="training_results.png")

def generate_images(
    training_run: str,
    outdir: str,
    figsize: tuple,
    dpi: int,
    name: str
):
    time.sleep(2)
    if not(outdir):
        outdir = training_run # save in same folder as training
    
    print(f'Creating plots for training run folder:"{training_run}"...')

    #Load data 
    root = './training-runs'
    json_filename = 'stats.jsonl'
    path = os.path.join(root, training_run, json_filename)
    stats = pd.read_json(path, lines=True)
    loss_scores_fake, loss_G, loss_D, r1_penalty, loss_age, hours = [], [], [], [], [], []
    loss_G_std, loss_D_std, loss_age_std = [],[],[]
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
        hours.append(data['Timing/total_hours']['mean'])
    
    fid = []
    fid_filename = 'metric-fid50k_full.jsonl'
    path_fid = os.path.join(root, training_run, fid_filename)
    fid_df = pd.read_json(path_fid, lines=True)
    for row in fid_df.iterrows():
        fid.append(row[1]["results"]["fid50k_full"])
    

    #convert to numpy
    loss_scores_fake, loss_G, loss_D, r1_penalty, loss_age, hours = np.array(loss_scores_fake), np.array(loss_G), np.array(loss_D), np.array(r1_penalty), np.array(loss_age), np.array(hours)
    loss_G_std, loss_D_std, loss_age_std = np.array(loss_G_std), np.array(loss_D_std), np.array(loss_age_std)

    fig, axs = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    x = range(len(hours))
    
    #GD_ylim = max(list(loss_G + loss_G_std) + list(loss_D + loss_D_std)) * 1.10
    ## GENERATOR LOSS ##
    axs[0,0].plot(x, loss_G, label="Loss")
    axs[0,0].set_ylabel("G loss")
    axs[0,0].fill_between(x, loss_G - loss_G_std, loss_G + loss_G_std, alpha=0.4, label="std")
    axs[0,0].set_ylim(0,max(loss_G + loss_G_std)*1.10)
    axs[0,0].legend()    

    ## DISCRIMINATOR LOSS ##
    axs[0,1].plot(x, loss_D, label="Loss")
    axs[0,1].set_ylabel("D loss")
    axs[0,1].set_ylim(0,max(loss_D + loss_D_std)*1.10)
    axs[0,1].fill_between(x, loss_D - loss_D_std, loss_D + loss_D_std, alpha=0.4, label="std")
    axs[0,1].legend()

    ## AGE LOSS ##
    axs[1,0].plot(x, loss_age, label="Loss")
    age_loss_ylim = max(loss_age + loss_age_std)
    axs[1,0].set_ylabel("Age loss")
    axs[1,0].set_ylim(0, age_loss_ylim*1.05)
    axs[1,0].fill_between(x, loss_age - loss_age_std, loss_age + loss_age_std, alpha=0.4, label="std")
    axs[1,0].legend()

    ## FID score
    axs[1,1].set_title("FID Score")
    axs[1,1].bar(range(len(fid)), fid, zorder=20)
    axs[1,1].set_xticks([])
    axs[1,1].grid(axis='y')
    
    
    # save figure
    plot_name = name
    plot_path = os.path.join(root, outdir, plot_name)
    plt.tight_layout()
    fig.savefig(plot_path)

    print("Ending program...")

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter