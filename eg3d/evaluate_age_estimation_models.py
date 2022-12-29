import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from Evaluation.fpage.fpage import FPAge
from training.estimate_age import AgeEstimatorNew
import torch 
from tqdm import tqdm 
import pandas as pd
from plot_training_results import plot_setup
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.data import Dataset
from pti_estimation import load
from time import time
class FaceDataset(Dataset):
    # from https://github.com/yu4u/age-estimation-pytorch/blob/65c268fbbbc4713c70ee55498bed8ceb9a4df56c/dataset.py#L35
    def __init__(self, data_dir, data_type, img_size=224, augment=False, age_stddev=1.0):
        assert(data_type in ("train", "valid", "test"))
        csv_path = Path(data_dir).joinpath(f"gt_avg_{data_type}.csv")
        img_dir = Path(data_dir).joinpath(data_type)
        self.img_size = img_size
        self.augment = augment
        self.age_stddev = age_stddev

        # if augment:
        #     self.transform = lambda i: i
        # else:
        #     self.transform = lambda i: i
        self.transform = lambda i: i
        self.x = []
        self.y = []
        self.std = []
        df = pd.read_csv(str(csv_path))
        ignore_path = Path(__file__).resolve().parent.joinpath("ignore_list.csv")
        ignore_img_names = list(pd.read_csv(str(ignore_path))["img_name"].values)

        for _, row in df.iterrows():
            img_name = row["file_name"]

            if img_name in ignore_img_names:
                continue

            img_path = img_dir.joinpath(img_name + "_face.jpg")
            assert(img_path.is_file())
            self.x.append(str(img_path))
            self.y.append(row["apparent_age_avg"])
            self.std.append(row["apparent_age_std"])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img_path = self.x[idx]
        age = self.y[idx]

        if self.augment:
            age += np.random.randn() * self.std[idx] * self.age_stddev

        img = cv2.imread(str(img_path), 1)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = self.transform(img).astype(np.float32)
        return torch.from_numpy(np.transpose(img, (2, 0, 1))), np.clip(round(age), 0, 100)

def load(image, return_tensor = True):
    image = cv2.imread(image) # load image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if return_tensor:
        return torch.from_numpy(image).float()
    else:
        return image
try:
    true_ages = np.load("true_ages.npy")
    fpage_hats = np.load("fpage_hats.npy")
    yu4u_hats = np.load("yu4u_hats.npy")
except:    
    dir = r"/work3/morbj/APPA/appa-real-release"
    test_dataset = FaceDataset(dir, "test", img_size=224, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False,
                                num_workers=8, drop_last=False)

    device = torch.device('cuda')
    yu4u = AgeEstimatorNew(device, crop=True)
    yu4u_no_crop = AgeEstimatorNew(device, crop=False)
    fpage = FPAge()

    root = r'/work3/morbj/UTK'
    true_ages = []; fpage_hats = []; yu4u_hats=[]; yu4u_hat_no_crops = []
    for x, y in tqdm(test_loader):
        x = x.to(device)
        y = y.to(device)
        yu4u_hat = yu4u.estimate_age_evaluate(x.to(device))
        true_ages = true_ages + y.tolist()
        yu4u_hats = yu4u_hats + yu4u_hat.tolist()
        for i, img in enumerate(x): #FPAge is not made to use batches it seems
            fpage_hat = fpage.estimate_age_rgb(img.permute(1,2,0))
            true_age = y[i].item()
            fpage_hats.append(fpage_hat)
            
    true_ages = np.array(true_ages)
    fpage_hats = np.array(fpage_hats)
    yu4u_hats = np.array(yu4u_hats)

    np.save("true_ages", true_ages)
    np.save("fpage_hats", fpage_hats)
    np.save("yu4u_hats", yu4u_hats)

fpage_no_detect = np.where(np.isnan(fpage_hats))[0]
print(f"FPAge could not detect and predict {sum(np.isnan(fpage_hats))} images")
fpage_hats_mae = np.mean(np.abs(np.delete(true_ages,fpage_no_detect) - np.delete(fpage_hats,fpage_no_detect)))
yu4u_hats_mae = np.mean(np.abs(true_ages - yu4u_hats))

df = pd.DataFrame._from_arrays(
    [true_ages, fpage_hats, yu4u_hats], 
    columns=["true_age", "fpage_hat", "yu4u_hat"], 
    index=list(range(len(true_ages)))
    )

df['yu4u_abs_error'] = np.abs(df.true_age - df.yu4u_hat)
df['fpage_abs_error'] = np.abs(df.true_age - df.fpage_hat)

plot_setup()
plt.figure(dpi=300, figsize=(6,4))
plt.subplot(2,1,1)
y, binEdges = np.histogram(df.true_age, bins=np.arange(1,max(df.true_age), 1))
bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
plt.plot(bincenters, y, label="True\ndistribution", alpha=0.7)

y, binEdges = np.histogram(df.fpage_hat, bins=np.arange(1,max(df.true_age), 1))
bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
plt.plot(bincenters[np.where(y!=0)[0]], y[np.where(y!=0)[0]], label="FPAge", alpha=0.7)

y, binEdges = np.histogram(df.yu4u_hat, bins=np.arange(1,max(df.true_age), 1))
bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
plt.plot(bincenters[np.where(y!=0)[0]], y[np.where(y!=0)[0]], label="yu4u", alpha=0.7)
ylim = plt.ylim()

print(f"FPAge MAE: {round(fpage_hats_mae, 3)}")
print(f"yu4u MAE: {round(yu4u_hats_mae, 3)}")
print("FPAge minimum:", df.fpage_hat[df.fpage_hat>0].min())
print("FPAge maximum:", df.fpage_hat.max())

print("yu4u minimum:", df.yu4u_hat.min())
print("yu4u maximum:", df.yu4u_hat.max())

print("True minimum:", df.true_age.min())
print("True maximum:", df.true_age.max())

print("Grouped by inverals")
print(df.groupby(pd.cut(df["true_age"], np.arange(0, 100, 5))).mean())
plt.ylim(ylim)
plt.xticks(np.arange(0,110,10))
plt.legend()
plt.xlabel("Age")
plt.ylabel("Count")

plt.subplot(2,1,2)
xticks_labels = [
    "(0, 5]","(5, 10]","(10, 15]","(15, 20]","(20, 25]","(25, 30]","(30, 35]","(35, 40]","(40, 45]","(45, 50]",
    "(50, 55]","(55, 60]","(60, 65]","(65, 70]","(70, 75]","(75, 80]","(80, 85]","(85, 90]"
]
ranges = np.arange(0, 95, 5)
number_of_ranges = len(ranges) - 1
x = range(number_of_ranges) # center the MAEs in the ranges
df_grouped_in_ranges = df.groupby(pd.cut(df["true_age"], ranges))
plt.plot(x, df_grouped_in_ranges.mean().yu4u_abs_error, label="yu4u")
plt.plot(x, df_grouped_in_ranges.mean().fpage_abs_error, label="FPAge")
plt.legend()
plt.xticks(x, xticks_labels, rotation=90)
plt.yticks(np.arange(0,30,5))
plt.xlabel("Age intervals")
plt.ylabel("MAE")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("appa-real.png", bbox_inches='tight')
plt.savefig("appa-real.pgf", bbox_inches='tight')


plt.figure(dpi=300, figsize=(6,2))
plt.plot(x, df_grouped_in_ranges.count().true_age)
plt.xticks(x, xticks_labels, rotation=90)
plt.xlabel("Age intervals")
plt.ylabel("\# images in\ninterval")
plt.grid(axis="y")
plt.ylim(0,400)
plt.yticks(np.arange(0,450,100))
plt.tight_layout()
plt.savefig("appa-real-count.png", bbox_inches='tight')
plt.savefig("appa-real-count.pgf", bbox_inches='tight')

# Evaluate effect of cropping on FFHQ

c = 0
root = r"/work3/morbj/FFHQ"
yu4u_no_crop = AgeEstimatorNew(torch.device("cuda"), age_min=0, age_max=75, crop=False)
yu4u_crop = AgeEstimatorNew(torch.device("cuda"), age_min=0, age_max=75, crop=True)
no_crop = []; crops = []
images = 0
no_crop_time = 0
crop_time = 0
for i, folder in tqdm(enumerate(os.listdir(root))):
    if folder == "dataset.json":
        continue
    if i > 3:
        break
    folder_path = os.path.join(root, folder)
    for image in tqdm(os.listdir(folder_path)):
        images += 1
        image_path = os.path.join(folder_path, image)
        image = load(image_path)
        tik = time()
        yu4u_hat_no_crop = yu4u_no_crop.estimate_age_rgb(image[None,:,:,:], normalize=False)
        tok = time()
        no_crop_time += (tok - tik)

        tik = time()
        yu4u_hat_crop = yu4u_crop.estimate_age_rgb(image[None,:,:,:], normalize=False)
        tok = time()
        crop_time += (tok - tik)

        no_crop.append(yu4u_hat_no_crop.item())
        crops.append(yu4u_hat_crop.item())


no_crop = np.array(no_crop)
crops = np.array(crops)
error = np.abs(no_crop - crops)
MAE = np.mean(np.abs(no_crop-crops))
print(f"Mean absolute difference in prediction with and without cropping is {round(MAE,3)}")
print(f"Tested on {images} images")

counts, bins = np.histogram(error, bins=100)
plt.figure(dpi=300, figsize=(5,3.5))
plt.hist(bins[:-1], bins, weights=np.log(counts+1))
# plt.yscale('log', base=2, nonposy='mask')
plt.xlabel("MAE")
t = np.concatenate((np.arange(0,10,1),np.arange(10,100,10), np.arange(100,1000,100),np.arange(1000,3000,1000)))
s=[""]*len(t)
s[0] = "0"; s[10]="10"; s[19]="100"; s[28]="1000"
plt.yticks(np.log(t+1), s)
plt.ylabel("Counts")
plt.savefig("cropping_error.png", bbox_inches='tight')
plt.savefig("cropping_error.pgf", bbox_inches='tight')