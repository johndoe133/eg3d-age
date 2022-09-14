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
home = os.path.expanduser('~')
sys.path.append(os.path.join(home, 'Documents/eg3d-age/eg3d/training'))
from estimate_age import AgeEstimator
import cv2


image_dir_name = 'image'

image_name = 'image'
use_multi_id_training = False
global_config.device = torch.device('cuda')
paths_config.e4e = './PTI/pretrained_models/e4e_ffhq_encode.pt'
paths_config.input_data_id = image_dir_name
paths_config.input_data_path = './PTI/image_processed'
paths_config.checkpoints_dir = './PTI/embeddings'
paths_config.style_clip_pretrained_mappers = './PTI/pretrained_models'
hyperparameters.use_locality_regularization = False

device = torch.device('cuda')

age=0.8

def run(age, model_path, image_name, run_pti_inversion):
    paths_config.stylegan2_ada_ffhq = model_path
    age_estimator = AgeEstimator()
    c = [0.9999064803123474, -0.006213949993252754, -0.012183905579149723, 0.028693876930960493, -0.0060052573680877686, -0.9998359084129333, 0.017090922221541405, -0.04020780808014847, -0.012288108468055725, -0.017016155645251274, -0.9997797012329102, 2.6995481091464293, 0.0, 0.0, 0.0, 1.0, 4.2647, 0.0, 0.5, 0.0, 4.2647, 0.5, 0.0, 0.0, 1.0] #, -0.4627]
    # c = np.array([0.9999064803123474, -0.006213949993252754, -0.012183905579149723, 0.028693876930960493, -0.0060052573680877686, -0.9998359084129333, 0.017090922221541405, -0.04020780808014847, -0.012288108468055725, -0.017016155645251274, -0.9997797012329102, 2.6995481091464293, 0.0, 0.0, 0.0, 1.0, 4.2647, 0.0, 0.5, 0.0, 4.2647, 0.5, 0.0, 0.0, 1.0, -0.4627])
    # c = [0.9984439611434937, -0.02431233786046505, -0.050184451043605804, 0.13143485450704356, -0.030380580574274063, -0.9918248057365417, -0.12393736839294434, 0.3216761509538541, -0.04676097631454468, 0.12526915967464447, -0.991020143032074, 2.6775453932526014, 0.0, 0.0, 0.0, 1.0, 4.2647, 0.0, 0.5, 0.0, 4.2647, 0.5, 0.0, 0.0, 1.0]
    image_path = os.path.join(paths_config.input_data_path, image_name + '.png')
    image = cv2.imread(image_path) # load image
    image =  cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image).float()
    estimated_age = age_estimator.estimate_age(image.permute(2,0,1)[None,:,:,:]) # so input shape is [1,3,512,512]
    c.append(estimated_age.item())
    
    c = np.reshape(c, (1, 26))
    c = torch.FloatTensor(c).cuda()
    if run_pti_inversion:
        print("Running PTI optimization...")
        model_id = run_PTI(c, image_name, use_wandb=False, use_multi_id_training=False)
        print("Finished running PTI optimization")
    return c

if __name__ == "__main__":
    run(age)
