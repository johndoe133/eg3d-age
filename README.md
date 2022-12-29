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
By default, training is done with a floating point age value, which also yields the best results. 
### Floating point age value
To train age-EG3D, you need to download and pre-process the dataset as described [here](https://github.com/NVlabs/eg3d#preparing-datasets:~:text=complete%20code%20example.-,Preparing%20datasets,-Datasets%20are%20stored). The `dataset.json` then needs to be renamed `dataset_MSE.json`. Each conditioning vector in `dataset_MSE.json` needs to be appended with the estimated age normalized to between -1 and 1. Example given in [evaluate_ages_wild.py](./eg3d/evaluate_ages_wild.py).

```
{"labels": [["00000/img00000000.png", [0.9422833919525146, 0.034289587289094925, 0.3330560326576233, -0.8367999667889383, 0.03984849900007248, -0.9991570711135864, -0.009871904738247395, 0.017018394869192363, 0.33243677020072937, 0.022573914378881454, -0.9428553581237793, 2.566997504832856, 0.0, 0.0, 0.0, 1.0, 4.2647, 0.0, 0.5, 0.0, 4.2647, 0.5, 0.0, 0.0, 1.0, -0.9545]], ...]}
```

### Categorical age values
Currently only supports 101 age categories. 


## Generate images

