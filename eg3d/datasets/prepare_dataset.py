
import os
import cv2 
import matplotlib.pyplot as plt
import numpy as np
import json

root = r"./datasets/FFHQ_512_6_balanced"
path = os.path.join(root, "dataset_ages.json")


ageconfig = r"./networks/age_model/age.prototxt"
agemodelpath = r"./networks/age_model/dex_chalearn_iccv2015.caffemodel"
age_model = cv2.dnn.readNetFromCaffe(ageconfig, agemodelpath)

def normalize(x, rmin = 5, rmax = 80, tmin = -1, tmax = 1):
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
    for i, face in enumerate(faces):
        image = face[0]
        c = face[1]
        age = normalize(c[-1])
        ages.append(age)
        c[-1] = age
        faces_copy.append([image, c])
        if i%10==0:
            print(i)
            print(age)
            print(image)
        
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

