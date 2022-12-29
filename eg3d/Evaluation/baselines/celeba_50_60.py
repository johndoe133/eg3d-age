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

@click.command()
@click.option('--images_path', help="Path of images to be evaluated", required=True)
def main(
    images_path: str,
):
    ages_path = '/zhome/d1/9/127646/Documents/eg3d-age/eg3d/Evaluation/baselines/ages_celeba.csv'
    df = pd.read_csv(ages_path)
    for image_name in tqdm(os.listdir(images_path)):
        new_image_name = image_name
        a = df.loc[df.image == new_image_name]
        estimated_age = a.fpage_hat.iloc[0]
        if not (estimated_age >= 50 and estimated_age <= 60):
            # change os.remove to print to test
            os.remove(os.path.join(images_path, image_name))
if __name__ == "__main__":
    main() 