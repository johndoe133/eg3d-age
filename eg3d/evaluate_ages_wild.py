
import os
import cv2 
import matplotlib.pyplot as plt
import numpy as np
import json
from training.estimate_age import AgeEstimatorNew
import torch
from tqdm import tqdm

root = r"/work3/s174379/datasets/FFHQ_all"
path = os.path.join(root, "dataset.json")



age_model = AgeEstimatorNew(torch.device('cuda:0'), age_max=75)

def normalize(x, rmin = 0, rmax = 100, tmin = -1, tmax = 1):
    z = ((x - rmin) / (rmax - rmin)) * (tmax - tmin) + tmin
    return round(z, 4)

def estimate_age(img):
    img_blob = cv2.dnn.blobFromImage(cv2.resize(img, (224, 224)))
    age_model.setInput(img_blob)
    age_dist = age_model.forward()[0]
    output_indexes = np.array([i for i in range(0, 101)])
    apparent_predictions = round(np.sum(age_dist * output_indexes), 2)
    return apparent_predictions

def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

ages = []
with open(path) as json_file:
    data = json.load(json_file)
    faces = np.array(data['labels'])
    data_copy = {"labels":[]}
    faces_copy = []
    for face in tqdm(faces):
        image_path = face[0]

        img = cv2.imread(os.path.join(root,image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_tensor = torch.from_numpy(img)
        
        predicted_age = age_model.estimate_age_rgb(img_tensor[None,:,:,:]).item()
        
        c = face[1]
        faces_copy.append([image_path, c[:-1] + [predicted_age]])
        
    data_copy["labels"] = faces_copy
    
# a = np.array(ages)
# print("Minimum estimated age:", a.min())
# print("Maximum estimated age:", a.max())
# print("Mean estimated age:", a.mean())  
# plt.hist(ages, bins=100, label = "Age distribution")
# plt.xlabel("Age")
# plt.ylabel("Counts")
# plt.title("Age distribution of data set")
# plt.savefig("age_distribution.png")
with open(os.path.join(root, "dataset_ages_2.json"), "w") as write_file:
    json.dump(data_copy, write_file)

