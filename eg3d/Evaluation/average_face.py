import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from tqdm import tqdm
import json
import imageio
import torch.nn.functional as F

def average_face(
    G, save_name, device, truncation_psi, truncation_cutoff, get_camera_parameters, get_conditioning_parameter, image_grid, network_folder
    ):
    """Aging effect on the *average face*. The average face is found by mapping 2000 latent codes from Z space to the W space
    for each age shown on the final image. The resulting w codes for each age is then averaged, ending up with one w code, which
    is the synthesized to a face image with the given age. 


    Args:
        G: generator
        save_name (str): name of save folder
        device (torch.device): 
        truncation_psi (float): 
        truncation_cutoff (int): 
        get_camera_parameters (function): returns the standard camera parameters for a given age
        get_conditioning_parameter (function): returns the standard conditioning parameters for a given age
        image_grid (function): to save image grid
        network_folder (str): folder of the trained network
    """

    training_option_path = os.path.join(network_folder, "training_options.json")
    f = open(training_option_path)
    training_option = json.load(f)
    age_min = training_option['age_min']
    age_max = training_option['age_max']

    font = ImageFont.truetype("FreeSerif.ttf", 72)
    text_color="#FFFFFF"

    save_path = os.path.join("Evaluation", "Runs", save_name)
    os.makedirs(save_path, exist_ok=True)

    np.random.seed(124)
    batch_size = 4
    faces_per_age = 2000 // batch_size

    images_age_min = -20
    images_age_max = 100
    ages = np.linspace(images_age_min, images_age_max, 12).round()
    age_loss_fn = "MSE"
    angle_y, angle_p = 0,0

    images_gif = []

    # Generate age in W space for each age
    for age in tqdm(ages):
        ws_list = []
        for i in range(faces_per_age):
            z = torch.from_numpy(np.random.randn(batch_size, G.z_dim)).to(device)
            c = get_conditioning_parameter(age, G, device, age_loss_fn, age_min, age_max)
            c = torch.repeat_interleave(c, batch_size, dim=0)
            ws = G.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
            ws_list.append(ws)

        w_avg = torch.stack(ws_list).mean(axis=[0,1])[None, :]
        c_camera = get_camera_parameters(age, G, device, angle_y, angle_p, age_loss_fn, age_min, age_max)
        
        generated_image =  G.synthesis(w_avg, c_camera)['image']
        
        img = (generated_image * 127.5 + 128).clamp(0, 255).to(torch.uint8)

        pil_img = Image.fromarray(img.permute(0,2,3,1)[0].cpu().numpy(), 'RGB')
        text_added = ImageDraw.Draw(pil_img)

        text_added.text((0,420), f"Age: {age}", font=font, fill=text_color)
        images_gif.append(pil_img)

    grid = image_grid(images_gif, rows=3, cols = 4)
    image_name = f"avg_face_{images_age_min}_{images_age_max}.png"
    grid.save(os.path.join(save_path, image_name))
    print(f"Saves {image_name} at {save_path}")

    # Linear interpolation
    rows = 5
    print("Generating z interpolation...")
    age1 = age_min
    age2 = age_max
    z1 = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)
    z2 = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)
    c_camera = get_camera_parameters(age, G, device, angle_y, angle_p, age_loss_fn, age_min, age_max)
    images_gif = []
    images = []
    for weight in np.linspace(0, 1, rows**2):
        z = torch.lerp(z1, z2, weight)
        age = age1 + age2 * weight
        c = get_conditioning_parameter(age, G, device, age_loss_fn, age_min, age_max)
        ws = G.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        generated_image =  G.synthesis(ws, c_camera)['image']
        img = (generated_image * 127.5 + 128).clamp(0, 255)
        img = F.interpolate(img.float(), [200,200],  mode='bilinear', align_corners=True).to(torch.uint8)
        pil_img = Image.fromarray(img.permute(0,2,3,1)[0].cpu().numpy(), 'RGB')
        images.append(pil_img)
        images_gif.append(np.array(pil_img))
    save = os.path.join(save_path, "z_interpolation")
    imageio.mimsave(save + ".gif", images_gif)
    grid = image_grid(images, rows = rows, cols = rows)
    grid.save(save + ".png")

    print("Generating w interpolation...")
    c1 = get_conditioning_parameter(age1, G, device, age_loss_fn, age_min, age_max)
    c2 = get_conditioning_parameter(age2, G, device, age_loss_fn, age_min, age_max)
    ws1 = G.mapping(z, c1, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
    ws2 = G.mapping(z, c2, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
    images_gif = []
    images = []
    for weight in np.linspace(0,1, rows**2):
        ws = torch.lerp(ws1, ws2, weight)
        generated_image =  G.synthesis(ws, c_camera)['image']
        img = (generated_image * 127.5 + 128).clamp(0, 255)
        img = F.interpolate(img.float(), [200,200],  mode='bilinear', align_corners=True).to(torch.uint8)
        pil_img = Image.fromarray(img.permute(0,2,3,1)[0].cpu().numpy(), 'RGB')
        images.append(pil_img)
        images_gif.append(np.array(pil_img))

    save = os.path.join(save_path, "w_interpolation")
    imageio.mimsave(save + ".gif", images_gif)
    grid = image_grid(images, rows = rows, cols = rows)
    grid.save(save + ".png")
        
    print("Done interpolating...")
    

if __name__ == "__main__":
    average_face()