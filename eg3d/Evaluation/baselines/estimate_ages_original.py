import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
home = str(os.path.expanduser('~'))
import sys
sys.path.append(os.path.join(home, 'Documents/eg3d-age/eg3d/training'))
from estimate_age import AgeEstimatorNew
import cv2
from tqdm import tqdm
import click
from Evaluation.fpage.fpage import FPAge
from training.face_id import FaceIDLoss
from pti_estimation import load
import matplotlib.pyplot as plt
from plot_training_results import plot_setup, compute_figsize


@click.command()
@click.option('--images_path', help="Path of images to be evaluated", required=False, default="/work3/s174379/datasets/FRGCv2-subset")
@click.option('--eval_ages', help="", required=False, default=True)
@click.option('--plot', help="", required=False, default=True)
@click.option('--celeba', help="", required=False, default=False)
def main(
    images_path: str,
    eval_ages: bool,
    plot: bool,
    celeba: bool,
):
    save_folder = "/zhome/d1/9/127646/Documents/eg3d-age/eg3d/Evaluation/baselines"
    if eval_ages:
        print("Loading fpage model")
        fpage = FPAge()
        columns = ["image", "fpage_hat"]
        res = []
        if not celeba:
            for image_name in tqdm(os.listdir(images_path)):
                image = load(os.path.join(images_path, image_name))
                fpage_hat = fpage.estimate_age_rgb(image)
                res.append([image_name, fpage_hat])
        else:
            print('males')
            for image_name in tqdm(os.listdir(os.path.join(images_path, "val", "male"))):
                image = load(os.path.join(images_path, "val", "male", image_name))
                fpage_hat = fpage.estimate_age_rgb(image)
                res.append([image_name, fpage_hat])
            print('females')
            for image_name in tqdm(os.listdir(os.path.join(images_path, "val", "female"))):
                image = load(os.path.join(images_path, "val", "female", image_name))
                fpage_hat = fpage.estimate_age_rgb(image)
                res.append([image_name, fpage_hat])
        df = pd.DataFrame(data=res, columns=columns)
        if celeba:
            df.to_csv(os.path.join(save_folder, "ages_celeba.csv"))
        else:
            df.to_csv(os.path.join(save_folder, "ages.csv"))
    else:
        if celeba:
            df = pd.read_csv(os.path.join(save_folder, "ages_celeba.csv"))
        else:
            df = pd.read_csv(os.path.join(save_folder, "ages.csv"))
    figsize = compute_figsize(400, 200)
    plot_setup()
    plt.figure(figsize=figsize, dpi=300)
    plt.title("Histogram of ages estimated by FPage of evaluation images")
    plt.hist(df["fpage_hat"], bins=10)
    fig_name = f"evaluation_histogram_fpage"
    save_path ="/zhome/d1/9/127646/Documents/eg3d-age/eg3d/Evaluation/baselines"
    plt.savefig(save_path + f"/{fig_name}" + ".png",bbox_inches='tight')
    plt.savefig(save_path + f"/{fig_name}" + ".pgf",bbox_inches='tight')
    print(f"Saved {fig_name} at {save_path}...")

if __name__ == "__main__":
    main()