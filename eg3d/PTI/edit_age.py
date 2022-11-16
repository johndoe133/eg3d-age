# from visualize_pti import load_generators
from configs import paths_config, hyperparameters, global_config
# from pti_pipeline import normalize
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import os
from PIL import Image, ImageFont, ImageDraw

def normalize(x, rmin = 5, rmax = 80, tmin = -1, tmax = 1):
    """Defined in eg3d.training.training_loop
    """
    z = ((x - rmin) / (rmax - rmin)) * (tmax - tmin) + tmin
    return np.round(z, 4)


def load_generators(image_name):
    with open(paths_config.stylegan2_ada_ffhq, 'rb') as f:
        old_G = pickle.load(f)['G_ema'].cuda()

    embedding_dir_G = './PTI/embeddings/G'
    with open(f'{embedding_dir_G}/{image_name}.pt', 'rb') as f_new: 
        new_G = torch.load(f_new).cuda()

    return old_G, new_G

def image_grid(imgs, rows, cols):
    #https://stackoverflow.com/questions/37921295/python-pil-image-make-3x3-grid-from-sequence-images
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


def edit_age(image_name, model_path, c):
    old_G, new_G = load_generators(image_name)
    ages = [age for age in np.linspace(-25,90,10)]
    embedding_dir_w = './PTI/embeddings/w'
    z_pivot = torch.load(f'{embedding_dir_w}/{image_name}.pt')
    images = []
    for age in ages:
        new_c = c
        new_c[0][-1] = normalize(age, rmin=hyperparameters.age_min, rmax=hyperparameters.age_max)
        w_pivot = new_G.mapping(z_pivot, c)
        new_image = new_G.synthesis(w_pivot, new_c)['image']
        img = (new_image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        img = Image.fromarray(img)
        # F.interpolate(img.float(), [224,224],  mode='bilinear', align_corners=True).shape
        images.append(img)
    rows = 2
    columns = 5
    grid = image_grid(images, rows, columns)
    home_dir = os.path.expanduser('~')
    path = f"Documents/eg3d-age/eg3d/PTI/output/{image_name}"
    draw = ImageDraw.Draw(grid)
    font_size = 80
    font = ImageFont.truetype("FreeSerif.ttf", font_size)
    counter = 0
    for i in range(rows):
        for j in range(columns):
            draw.text((j*512 , i*512 + 500 - font_size), f"Age: {int(ages[counter])}", (255,255,255), font=font)
            counter += 1

    save_name = os.path.join(home_dir, path, "aging_effect.png")
    grid.save(save_name)
    return img
