import os
import sys
import pickle
import numpy as np
from PIL import Image
import torch
from configs import paths_config, hyperparameters, global_config
from utils.align_data import pre_process_images
from scripts.run_pti import run_PTI
import matplotlib.pyplot as plt
from scripts.latent_editor_wrapper import LatentEditorWrapper
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
import sys
import json
home = os.path.expanduser('~')
sys.path.append(os.path.join(home, 'Documents/eg3d-age/eg3d/training'))
from estimate_age import AgeEstimator, AgeEstimatorNew
import cv2


image_dir_name = 'image'

image_name = 'image'
use_multi_id_training = False
global_config.device = torch.device('cuda')
paths_config.e4e = './PTI/pretrained_models/e4e_ffhq_encode.pt'
paths_config.input_data_id = image_dir_name
paths_config.checkpoints_dir = './PTI/embeddings'
paths_config.style_clip_pretrained_mappers = './PTI/pretrained_models'
hyperparameters.use_locality_regularization = False

device = torch.device('cuda')

def denormalize(z, rmin = 5, rmax = 80, tmin = -1, tmax = 1):
    """Cant import from training.training_loop due to naming of folders...
    """
    x = (z*(rmax - rmin)- tmin*(rmax-rmin))/(tmax-tmin)+rmin
    return x

def run(model_path, image_name, run_pti_inversion):
    #Load models
    paths_config.stylegan2_ada_ffhq = model_path
    # age_estimator = AgeEstimator()
    age_estimator = AgeEstimatorNew(torch.device("cuda"))

    #Load pose parameters 
    home = os.path.expanduser('~')
    json_path = r"Documents/eg3d-age/dataset_preprocessing/ffhq/Deep3DFaceRecon_pytorch/checkpoints/pretrained/results/test/epoch_20_000000/dataset.json"
    c_path = os.path.join(home, json_path)
    f = open(c_path)
    data = json.load(f)
    dictionary = dict(data['labels'])
    c = dictionary[image_name + ".jpg"]

    # Add age label
    image_path = os.path.join(paths_config.input_data_path, image_name + '.png')
    image = cv2.imread(image_path) # load image
    image =  cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image).float()
    estimated_age = age_estimator.estimate_age_rgb(image[None,:,:,:]) # so input shape is [1,3,512,512]
    c.append(estimated_age.item())
    print("######################################")
    print("Estimated age is", denormalize(estimated_age).item())
    print("######################################")
    c = np.reshape(c, (1, 26))
    c = torch.FloatTensor(c).cuda()
    
    if run_pti_inversion:
        print("Running PTI optimization...")
        model_id = run_PTI(c, image_name, use_wandb=False, use_multi_id_training=False)
        print("Finished running PTI optimization")
    return c
