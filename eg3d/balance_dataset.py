import numpy as np
import matplotlib.pyplot as plt
import json
import time
import shutil

time.sleep(3)

def flatten(l):
    return [item for sublist in l for item in sublist]

f = open('./datasets/FFHQ_512_6_balanced/dataset_ages.json')
data = json.load(f)

new_data = []

bins_vals = [10,20,30,40,50,60,80]
bins_paths = [[],[],[],[],[],[]]
bins_params = [[],[],[],[],[],[]]

# reflected
bins_paths_2 = [[],[],[],[],[],[]]
bins_params_2 = [[],[],[],[],[],[]]

for index in range(len(data['labels'])):
    if index % 2 == 0:
        item = data['labels'][index]
        age = item[1][-1]
        path = item[0]

        item_2 = data['labels'][index+1] # reflected
        age_2 = item_2[1][-1] # reflected
        path_2 = item_2[0] # reflected
        for i in range(len(bins_vals[:-1])):
            lower = bins_vals[i]
            upper = bins_vals[i+1]
            if age >= lower and age <= upper:
                bins_paths[i].append(path)
                bins_params[i].append(item[1])
                bins_paths_2[i].append(path_2) # reflected
                bins_params_2[i].append(item_2[1]) # reflected

    
all_lengths = []
for item in bins_paths:
    all_lengths.append(len(item))

desired_size = min(all_lengths)
bins_paths_new = [[],[],[],[],[],[]]
bins_params_new = [[],[],[],[],[],[]]
for index in range(len(bins_paths)):
    bins_paths_new[index] = bins_paths[index][:desired_size] + bins_paths_2[index][:desired_size]
    bins_params_new[index] = bins_params[index][:desired_size] + bins_params_2[index][:desired_size]


paths_new = flatten(bins_paths_new)
params_new = flatten(bins_params_new)
data['labels'] = []
print('copying files over')
for path, param in zip(paths_new, params_new):
    data['labels'].append([path,param])
    # shutil.copyfile('datasets/FFHQ_512_6/' + path, 'datasets/FFHQ_512_6_balanced/' + path)

with open("datasets/FFHQ_512_6_balanced/dataset.json", "w") as outfile:
    json.dump(data, outfile)