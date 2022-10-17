
import os
import cv2 
import matplotlib.pyplot as plt
import numpy as np
import json
from tqdm import tqdm
import time
import sys
path =  "/zhome/d7/6/127158/Documents/eg3d-age/eg3d"
sys.path.append(path)
from training.estimate_age import AgeEstimatorNew
from training.training_loop import normalize, denormalize
import torch

root = r"datasets/FFHQ_512_6_balanced"
path = os.path.join(root, "dataset_norm.json")

# age_model = AgeEstimatorNew(torch.device("cuda"))

def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

ages = []
with open(path) as json_file:
    data = json.load(json_file)
    faces = data['labels']
    data_copy = {"labels":[]}
    faces_copy = []
    for face in tqdm(faces):
        image = face[0]
        c = face[1]
        age = c[-1]
        age = denormalize(age)
        l = [0] * 101
        l[int(age)] = 1
        c = c[:-1] + l
        ages.append(age)
        faces_copy.append([image, c])
        
    data_copy["labels"] = faces_copy
    
a = np.array(ages)
print("Minimum estimated age:", a.min())
print("Maximum estimated age:", a.max())
print("Mean estimated age:", a.mean())  
plt.hist(ages, bins=100, label = "Age distribution")
plt.xlabel("Age")
plt.ylabel("Counts")
plt.title("Age distribution of data set")
plt.savefig("age_distribution.png")
with open(os.path.join(root, "dataset-copy.json"), "w") as write_file:
    json.dump(data_copy, write_file)

