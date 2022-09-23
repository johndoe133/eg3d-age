import os
import click
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import pandas as pd
import json


def compute_figsize(pt_x, pt_y):
    px = 1/72  # pixel in inches'
    width = pt_x * px
    height = pt_y * px
    return (width, height)

@click.command()
@click.option('--training_run', type=str, help='Past in folder name in training run', required=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=False)
@click.option('--width', help='Width of the image', type=int, default=426)
@click.option('--height', help='Width of the image', type=int, default=600)
@click.option('--dpi', help='Dots per image for image', type=int, default=300)
@click.option('--name', help='Name of the output image', type=str, default="training_results.png")
@click.option('--pgfname', help='Name of the output tex image file', type=str, default="training_results.pgf")
@click.option('--id', help='If id loss is available', type=bool, default=True)
def generate_images(
    training_run: str,
    outdir: str,
    width: int,
    height: int,
    dpi: int,
    name: str,
    pgfname: str,
    id: bool,
):
    if not(outdir):
        outdir = training_run # save in same folder as training
    
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })

    

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
    snap = training_option['network_snapshot_ticks']
    f.close()

    #convert to numpy
    loss_scores_fake, loss_G, loss_D, r1_penalty, loss_age, loss_id, hours = np.array(loss_scores_fake), np.array(loss_G), np.array(loss_D), np.array(r1_penalty), np.array(loss_age), np.array(loss_id), np.array(hours)
    loss_G_std, loss_D_std, loss_age_std, loss_id_std = np.array(loss_G_std), np.array(loss_D_std), np.array(loss_age_std), np.array(loss_id_std)

    figsize = compute_figsize(width, height)
    fig, axs = plt.subplots(4, 1, sharex=False, figsize = figsize, dpi=dpi)
    (axs1, axs2, axs3, axs4) = axs.ravel()
    img_pr_tick = 4000
    x = range(len(hours))
    x = np.array(x) * img_pr_tick
    
    #GD_ylim = max(list(loss_G + loss_G_std) + list(loss_D + loss_D_std)) * 1.10
    ## AGE LOSS ##
    axs1.plot(x, loss_age, label="Loss")
    age_loss_ylim = max(loss_age + loss_age_std)
    axs1.set_ylabel("$\mathcal{L}_{Age}$")
    axs1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x/1000) + 'k'))
    axs1.set_ylim(0, age_loss_ylim*1.05)
    axs1.fill_between(x, loss_age - loss_age_std, loss_age + loss_age_std, alpha=0.4, label="std")
    axs1.set_title("With $age_{scale}$ = " + str(age_scale))
    axs1.legend(loc='upper right')

    ## ID LOSS ##
    axs2.plot(x, loss_id, label="Loss")
    if id:
        id_loss_ylim = max(loss_id + loss_id_std)
    else:
        id_loss_ylim = 1
    axs2.set_ylabel("$\mathcal{L}_{ID}$")
    axs2.set_ylim(0, id_loss_ylim*1.05)
    axs2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x/1000) + 'k'))
    axs2.fill_between(x, loss_id - loss_id_std, loss_id + loss_id_std, alpha=0.4, label="std")
    axs2.set_title("With $id_{scale}$ = " + str(id_scale))
    axs2.legend(loc='upper right')

    ## GENERATOR LOSS ##
    axs3.set_ylabel("$\mathcal{L}_{G}$")
    axs3.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x/1000) + 'k'))
    axs3.fill_between(x, loss_G - loss_G_std, loss_G + loss_G_std, alpha=0.4, label="std")
    axs3.set_ylim(0,max(loss_G + loss_G_std)*1.10)
    axs3.plot(x, loss_G, label="Loss")
    axs3.legend(loc='upper right')    

    ## DISCRIMINATOR LOSS ##
    axs4.plot(x, loss_D, label="Loss")
    axs4.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x/1000) + 'k'))
    axs4.set_ylabel("$\mathcal{L}_{D}$")
    axs4.set_xlabel("Images trained on")
    axs4.set_ylim(0,max(loss_D + loss_D_std)*1.10)
    axs4.fill_between(x, loss_D - loss_D_std, loss_D + loss_D_std, alpha=0.4, label="std")
    axs4.legend(loc='upper right')

    fig.subplots_adjust(hspace=0)

    # save figure
    plot_path = os.path.join(root, outdir, name)
    plot_path_pgf = os.path.join(root, outdir, pgfname)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.savefig(plot_path_pgf)
    
    print(f'Saved figures to {os.path.join(root, outdir)}')
    
    
    
    ## FID score
    print("Making FID plot...")
    plt.figure(figsize=(2.5,3), dpi=dpi)
    plt.ylabel("FID Score")
    x_fid = np.array(range(len(fid)))
    plt.bar(x_fid, fid, zorder=20)
    plt.xlabel("Images trained on")
    plt.xticks(x_fid, [str(int(x)) + 'k' for x in x_fid*img_pr_tick*snap/1000])
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(root, outdir, "fid_score.png"))
    plt.savefig(os.path.join(root, outdir, "fid_score.pgf"))
    print("Ending program...")

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter