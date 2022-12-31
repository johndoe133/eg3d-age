import json
import os
from tqdm import tqdm

def normalize(x, rmin = 0, rmax = 100, tmin = -1, tmax = 1):
    z = ((x - rmin) / (rmax - rmin)) * (tmax - tmin) + tmin
    return round(z, 4)

path = "/work3/morbj/FFHQ/dataset_mse.json"

f = open(path)
json_file = json.load(f)
f.close()


data = json_file['labels']
data_category = dict({'labels': []})
for image in tqdm(data):
    l = [0.0]*101
    age = image[1][-1]
    age = int(normalize(age, rmin=-1, rmax=1, tmin=0, tmax=75))
    l[age] = 1.0
    data_category['labels'] += [[image[0], image[1][:-1] + l]]

save_dir = os.path.join("/work3/morbj/FFHQ", "dataset_cat.json")
print(f'saving file at {save_dir}')
out_file = open(save_dir, "w")
json.dump(data_category, out_file)
