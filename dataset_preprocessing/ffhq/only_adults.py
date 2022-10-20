#!/usr/bin/python
from PIL import Image
import os
import sys
import json
from tqdm import tqdm

os.chdir('/work3/s174379/datasets/')

path = "FFHQ_512_6/"

adult_path = "FFHQ_512_18/"

def normalize(x, rmin = 0, rmax = 100, tmin = -1, tmax = 1):
    z = ((x - rmin) / (rmax - rmin)) * (tmax - tmin) + tmin
    return round(z, 4)

dirs = os.listdir( path )

f = open(os.path.join(path, 'dataset_ages_0_100_norm.json'))
data = json.load(f)['labels']
data_adults = dict({'labels': []})
os.chdir(os.path.join('/work3/s174379/datasets/'))
for image in tqdm(data):
    age = image[1][-1]
    if age >= -0.6:
        age = normalize(age, rmin=-0.6, rmax=1)
        image[1][-1] = age
        data_adults['labels'] += [image]
        src_path = os.path.join(path, image[0])
        dst_path = os.path.join(adult_path, image[0].split('/')[0])
        os.system(f'cp {src_path} {dst_path}')

f = open(os.path.join(adult_path, 'dataset_mse.json'), 'w')
json.dump(data_adults, f)