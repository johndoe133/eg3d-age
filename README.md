# Face Age Progression Based on Generative Adversarial Networks

**Thesis by [Eric Jensen](https://www.linkedin.com/in/erickastljensen/) and [Morten Bjerre](https://www.linkedin.com/in/morten-bjerre/).**

## Conditioning [EG3D](https://github.com/NVlabs/eg3d) on age

![Image](./eg3d/example.gif)

### Abstract
INSERT ABSTRACT

![Image](./eg3d/example.png)

## Installation
In order to replicated age-EG3D, first clone the repository and change the directory

```
git clone https://github.com/johndoe133/eg3d-age/
cd eg3d-age/eg3d
```

Then, use conda to create the environment and activate it

```
conda env create -f ./environments/eg3d.yml
conda activate eg3d
```

Then install the `environments/requirements.txt` file

```
pip install -r ./environments/requirements.txt 
```

Age-eg3d runs with a special version of pytorch and torchvision which (to our knowledge) cannot be installed with `conda`. It has to be installed with `pip`. Installing the environment will, however, automatically install pytorch and torchvision because of dependencies. Therefore, first make sure that pytorch and torchvision is uninstalled and not showing when running `conda list` and `pip list`. 

To install the correct version of pytorch and torchvision run
```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

## Training
### Setup
To train age-EG3D, you need to download and pre-process the dataset as described [here](https://github.com/NVlabs/eg3d#preparing-datasets:~:text=complete%20code%20example.-,Preparing%20datasets,-Datasets%20are%20stored).

By default, training is done with a floating point age value, which also yields the best results. 
#### Floating point age value
 The `dataset.json` needs to be renamed `dataset_MSE.json`. Each conditioning vector in `dataset_MSE.json` needs to be appended with the estimated age normalized to between -1 and 1. Example of this given in [evaluate_ages_wild.py](./eg3d/evaluate_ages_wild.py). The json file should look similar to the following:

```
{"labels": [["00000/img00000000.png", [0.9422833919525146, 0.034289587289094925, 0.3330560326576233, -0.8367999667889383, 0.03984849900007248, -0.9991570711135864, -0.009871904738247395, 0.017018394869192363, 0.33243677020072937, 0.022573914378881454, -0.9428553581237793, 2.566997504832856, 0.0, 0.0, 0.0, 1.0, 4.2647, 0.0, 0.5, 0.0, 4.2647, 0.5, 0.0, 0.0, 1.0, -0.9545]], ...]}
```

#### Categorical age values
Age-EG3D currently only supports using a categorical condtioning vector of size 101. The `dataset.json` needs to be renamed `dataset_MSE.json`. The file then needs to be preprocessed like shown in [create_categories_json.py](./eg3d/create_categories_json.py).

### Train the model
The model can be trained by running [train.sh](./eg3d/Experiments/train.sh):
```
sh Experiments/train.sh
```
The meaning of the argmunts for training is described in [train.py](./eg3d/train.py).

### Pre-trained model
The optimal model of the age-EG3D thesis can be downloaded [here](https://drive.google.com/file/d/1sLwxKsOPUjtZkT66VoPc9kjt8VtbAj3q/view?usp=sharing). 

## Generate images
Use the shell script [generate_images.sh](./eg3d/Experiments/generate_images.sh) to generate an image and a gif file. This is easily done by adding `--network_folder=PATH TO TRAINING FOLDER` after having trained a model. An example could be  `--network_folder=./training-runs/00001/00000-ffhq-FFHQ-gpus2-batch8-gamma5`. Otherwise use the following snippet with `network_pkl='PATH TO TRAINED MODEL'`:
```
def normalize(x, rmin = 0, rmax = 75, tmin = -1, tmax = 1):
    # normalize age between -1 and 1
    z = ((x - rmin) / (rmax - rmin)) * (tmax - tmin) + tmin
    return z

device = torch.device('cuda')
with dnnlib.util.open_url(network_pkl) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)

z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

fov_deg = 18.837
intrinsics = FOV_to_intrinsics(fov_deg, device=device)
cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0,0,0]), device=device)
cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)    

age = TARGET AGE
age = [normalize(age, rmin=age_min, rmax=age_max)]
c = torch.cat((conditioning_params, torch.tensor([age], device=device)), 1)
c_params = torch.cat((camera_params, torch.tensor([age], device=device)), 1)
ws = G.mapping(z, c.float(), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
img = G.synthesis(ws, c_params.float())['image']
img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
pil_img = Image.fromarray(img[0].cpu().numpy(), 'RGB')
```
