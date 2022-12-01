# Invert every image in /zhome/d7/6/127158/Documents/eg3d-age/dataset_preprocessing/ffhq/Deep3DFaceRecon_pytorch/test/crop

# Save everything in work3

# Create folder in Evaluation/Inversion/{network}/{image_name}/age0.png, age10.png

import os
import sys
import pickle
import numpy as np
from PIL import Image
import torch
from configs import paths_config, hyperparameters, global_config
from scripts.run_pti import run_PTI
import matplotlib.pyplot as plt
import sys
import json
home = os.path.expanduser('~')
sys.path.append(os.path.join(home, 'Documents/eg3d-age/eg3d/training'))
from estimate_age import AgeEstimatorNew
import cv2
from tqdm import tqdm
import click

@click.command()
@click.option('--model_path', help="Relative path to the model", required=True)
@click.option('--trunc', help="Truncation psi in inversion", required=False, default=1.0)
def main(
        model_path: str,
        trunc: float,
    ):
    save_name = f"{model_path.split('/')[-2][:5]}-{model_path.split('/')[-1].split('.')[0][8:]}-trunc-{trunc}"
    
    paths_config.checkpoints_dir = f'/work3/morbj/embeddings/{save_name}'
    device = torch.device('cuda')
    image_folder = '/zhome/d7/6/127158/Documents/eg3d-age/dataset_preprocessing/ffhq/Deep3DFaceRecon_pytorch/test/crop'
    home = os.path.expanduser('~')
    json_path = r"Documents/eg3d-age/dataset_preprocessing/ffhq/Deep3DFaceRecon_pytorch/checkpoints/pretrained/results/test/epoch_20_000000/dataset.json"
    c_path = os.path.join(home, json_path)
    age_estimator = AgeEstimatorNew(torch.device("cuda"), age_min=hyperparameters.age_min, age_max=hyperparameters.age_max)
    paths_config.stylegan2_ada_ffhq = model_path
    # Invert every image
    pbar = tqdm(os.listdir(image_folder))
    for image_name in pbar:
        embeddings_G = os.path.join(paths_config.checkpoints_dir, "G")     
        if image_name.replace("png", "pt") in os.listdir(embeddings_G):
            continue #image already inverted
        pbar.set_description(f"Processing {image_name}")

        ## Use estimated age as starting point for age label in inversion
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path) # load image
        image =  cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).float()
        estimated_age = age_estimator.estimate_age_rgb(image[None,:,:,:]) # so input shape is [1,3,512,512]
        
        image_name = image_name.strip('.png')
        f = open(c_path)
        data = json.load(f)
        dictionary = dict(data['labels'])
        c = dictionary[image_name + ".jpg"]
        f.close()

        c.append(estimated_age.item())
        c = np.reshape(c, (1, 26))
        c = torch.FloatTensor(c).cuda()
        
        run_PTI(c, image_name, use_wandb=False, use_multi_id_training=False, evaluation=True, trunc=trunc)

    # Generate image of ages [0,10,20,30,...,100]
    ages = np.arange(0,110,10)
    for image_name in tqdm(os.listdir(image_folder)):
        image_name = image_name.strip('.png')
        # create folder
        pbar.set_description(f"Saving images for {image_name}")
        folder_name = os.path.join(paths_config.checkpoints_dir, image_name)
        os.makedirs(folder_name, exist_ok=True)

        # load PTI G and z pivot
        new_G, z_pivot= load_generator(image_name)

        # prepare c
        f = open(c_path)
        data = json.load(f)
        dictionary = dict(data['labels'])
        c = dictionary[image_name + ".jpg"]
        f.close()
        c.append(0) # changed later
        c = np.reshape(c, (1, 26))
        c = torch.FloatTensor(c).cuda()

        # generate images of different ages
        for age in ages:
            new_c = c
            new_c[0][-1] = normalize(age, rmin=hyperparameters.age_min, rmax=hyperparameters.age_max)
            w_pivot = new_G.mapping(z_pivot, c, truncation_psi = trunc)
            new_image = new_G.synthesis(w_pivot, new_c)['image']
            img = (new_image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            img = Image.fromarray(img)
            img.save(f"{folder_name}/{age}.png")

def normalize(x, rmin = 0, rmax = 75, tmin = -1, tmax = 1):
    """Defined in eg3d.training.training_loop
    """
    z = ((x - rmin) / (rmax - rmin)) * (tmax - tmin) + tmin
    return np.round(z, 4)


def load_generator(image_name):
    with open(f'{paths_config.checkpoints_dir}/G/{image_name}.pt', 'rb') as f_new: 
        new_G = torch.load(f_new).cuda()
    z_pivot = torch.load(f'{paths_config.checkpoints_dir}/w/{image_name}.pt')
    return new_G, z_pivot
    
if __name__ == "__main__":
    main()