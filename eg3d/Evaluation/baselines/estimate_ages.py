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
@click.option('--original_images', help="Path of original un-aged images", required=False, default="/work3/s174379/datasets/FRGCv2-aligned")
@click.option('--age_model', help="Age model name", required=True)
@click.option('--restart', help="delete old baselines.csv and make new one", required=False, default=False)
def main(
    images_path: str,
    original_images: str,
    age_model: str,
    restart: bool,
):
    baselines_path = '/zhome/d1/9/127646/Documents/eg3d-age/eg3d/Evaluation/baselines/baselines.csv'
    cosine_sim_f = torch.nn.CosineSimilarity()
    print("Loading fpage model")
    fpage = FPAge()
    device = torch.device("cuda")
    print("Loading yu4u model")
    yu4u = AgeEstimatorNew(torch.device("cuda"))
    arcface = FaceIDLoss(device, model = "ArcFace")

    # get arcface vectors of all original faces
    print('Getting arcface of original images')
    arcface_original = {}
    for image_name in tqdm(os.listdir(original_images)):
        image = load(os.path.join(original_images, image_name))
        arcface_original[image_name] = arcface.get_feature_vector_arcface(image.numpy())


    print('getting cosine similarities and age estimations')
    columns = ["age_model", "image_name", "target_age", "age_hat_yu4u", "age_hat_fpage", "cos_sim"]
    res = []
    for dir in tqdm(os.listdir(images_path)):
        target_age = int(dir)
        for image_name in os.listdir(os.path.join(images_path, dir)):
            image = load(os.path.join(images_path, dir, image_name))
            fpage_hat = fpage.estimate_age_rgb(image)
            yu4u_hat = yu4u.estimate_age_rgb(image[None,:,:,:], normalize=False)
            f2 = arcface.get_feature_vector_arcface(image.numpy())

            # HRFAE saves image names with age in them and as .jpg instead of .png
            if '_age_' in image_name:
                image_name = image_name.split('_age_')[0] + '.png'
            cos_sim = cosine_sim_f(arcface_original[image_name], f2).item()
            res.append([age_model, image_name, target_age, yu4u_hat.item(), fpage_hat, cos_sim])
    if os.path.isfile(baselines_path) and not restart:
        df = pd.read_csv(baselines_path)
        df = df.append(pd.DataFrame(res, columns=columns), ignore_index=True)
    else:
        df = pd.DataFrame(data=res, columns=columns)
    save_folder = "/zhome/d1/9/127646/Documents/eg3d-age/eg3d/Evaluation/baselines"
    df.to_csv(os.path.join(save_folder, "baselines.csv"))

if __name__ == "__main__":
    main() 