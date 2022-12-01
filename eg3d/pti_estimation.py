import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
# home = os.path.expanduser('~')
# sys.path.append(os.path.join(home, 'Documents/eg3d-age/eg3d/training'))
from training.estimate_age import AgeEstimatorNew
import cv2
from tqdm import tqdm
import click
from Evaluation.fpage.fpage import FPAge
from training.face_id import FaceIDLoss
@click.command()
@click.option('--model_path', help="Relative path to the model", required=True)
@click.option('--trunc', help="Truncation psi in inversion", required=False, default=1.0)
@click.option('--age_min', help="Minimum age during training", required=False, default=0)
@click.option('--age_max', help="Maximum age during training", required=False, default=75)
def main(
        model_path: str,
        trunc: float,
        age_min: int,
        age_max: int,

    ):
    save_name = f"{model_path.split('/')[-3]}-{model_path.split('/')[-1].split('.')[0][8:]}-trunc-{trunc}"
    
    root = f'/work3/morbj/embeddings/{save_name}'
    device = torch.device('cuda')
    og_image_folder = '/zhome/d7/6/127158/Documents/eg3d-age/dataset_preprocessing/ffhq/Deep3DFaceRecon_pytorch/test/crop'
    home = os.path.expanduser('~')
    yu4u = AgeEstimatorNew(torch.device("cuda"), age_min=age_min, age_max=age_max)
    fpage = FPAge()
    arcface = FaceIDLoss(device, model = "ArcFace")
    columns = ["image_name", "target_age", "age_hat_yu4u", "age_hat_fpage", "cos_sim"]
    res = []
    cosine_sim_f = torch.nn.CosineSimilarity()
    for image_folder in tqdm(os.listdir(root)): #the folder happens to be the name of the original image
        if image_folder in ['G', 'w']:
            continue
        
        orginal_image_path = os.path.join(og_image_folder, image_folder + ".png")
        original_image = load(orginal_image_path, return_tensor=False)
        f1 = arcface.get_feature_vector_arcface(original_image)
        folder = os.path.join(root, image_folder)
        for image_name in os.listdir(folder):
            target_age = int(image_name.strip(".png"))
            image_path = os.path.join(folder, image_name)
            image = load(image_path)
            yu4u_hat = yu4u.estimate_age_rgb(image[None,:,:,:], normalize=False) # so input shape is [1,3,512,512]
            fpage_hat = fpage.estimate_age_rgb(image)
            f2 = arcface.get_feature_vector_arcface(image.numpy())
            cos_sim = cosine_sim_f(f1, f2)
            res.append([image_folder, target_age, yu4u_hat.item(), fpage_hat, cos_sim.item()])
    
    df = pd.DataFrame(data=res, columns=columns)   
    save_folder = os.path.join("Evaluation", "Runs", save_name)
    os.makedirs(save_folder, exist_ok=True)
    df.to_csv(os.path.join(save_folder, "pti.csv"))

def load(image, return_tensor = True):
    image = cv2.imread(image) # load image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if return_tensor:
        return torch.from_numpy(image).float()
    else:
        return image
            
def normalize(x, rmin = 0, rmax = 75, tmin = -1, tmax = 1):
    """Defined in eg3d.training.training_loop
    """
    z = ((x - rmin) / (rmax - rmin)) * (tmax - tmin) + tmin
    return np.round(z, 4)
    
if __name__ == "__main__":
    main()