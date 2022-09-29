import os
import torch
import numpy as np
import torch.nn.functional as F
import sys
import os
import torch.nn as nn
import pretrainedmodels
import pretrainedmodels.utils
home = os.path.expanduser('~')
path = os.path.join(home, "Documents/eg3d-age/eg3d/")
sys.path.append(path)
from networks.dex.models import Age
from networks.age_estimation.defaults import _C as cfg
import os 
import dlib
import cv2

class AgeEstimatorNew():
    def __init__(self, device):
        root = os.path.expanduser('~')
        model = get_model(model_name=cfg.MODEL.ARCH, pretrained=None)
        self.age_model = model.to(device)
        pretrained_model_path = os.path.join(root, "Documents/eg3d-age/eg3d/networks/yu4u.pth")
        checkpoint = torch.load(pretrained_model_path)#, map_location="cpu")
        self.age_model.load_state_dict(checkpoint['state_dict'])
        self.age_model.requires_grad_(requires_grad=False) 
        self.age_model.eval()
        self.device = device
        self.margin = 0.4
        self.detector = dlib.get_frontal_face_detector()
        self.img_size = 224
        self.ages = torch.arange(0,101, device = self.device)
    
    def estimate_age(self, gen_img, normalize = True):
        """Takes output of G.synthesis and estimates the age of the synthetic person.

        Args:
            gen_img (tensor): image

        Returns:
            tensor: age
        """
        img = gen_img['image']
        img_RGB = (img * 127.5 + 128).clamp(0, 255)
        img_RGB = img_RGB.permute(0,2,3,1) # to fit detector input shape
        img_RGB_detached = img_RGB.detach()
        detections = self.detect_faces(img_RGB_detached)
        crops = self.crop_images(detections, img_RGB)
        crops = crops.to(self.device)
        outputs = F.softmax(self.age_model(crops), dim=-1)
        predicted_ages = (outputs * self.ages).sum(axis=-1)
        if normalize:
            return self.normalize_ages(predicted_ages)
        else:
            return predicted_ages

    def estimate_age_rgb(self, image, normalize = True):
        #image = image.permute(0,2,3,1)
        detections = self.detect_faces(image)
        image = image.type('torch.FloatTensor')
        crops = self.crop_images(detections, image)
        crops = crops.to(self.device)
        outputs = F.softmax(self.age_model(crops), dim=-1)
        predicted_ages = (outputs * self.ages).sum(axis=-1)
        if normalize:
            return self.normalize_ages(predicted_ages)
        else:
            return predicted_ages

    def detect_faces(self, img_RGB):
        detections = [] # x1, y1, x2, y2, w, h 
        for image in img_RGB: # iterate over the batch
            width, height = image.shape[0], image.shape[0]
            image = image.to(torch.uint8)
            image = image.cpu().numpy()
            detected = self.detector(image, 1)
            if len(detected) == 0:
                # no face was found - use the whole image
                x1, y1, x2, y2, w, h = 0, 0, width, height, width, height
            else:
                d = detected[0]
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            detections.append([x1, y1, x2, y2, w, h])
        return detections

    def crop_images(self, detections, img_RGB):
        batch_size, img_w, img_h, channels = img_RGB.shape[0],img_RGB.shape[1],img_RGB.shape[2],img_RGB.shape[3]
        cropped = torch.empty(batch_size, channels, self.img_size, self.img_size)
        for i in range(batch_size):
            x1, y1, x2, y2, w, h = detections[0]
            xw1 = max(int(x1 - self.margin * w), 0)
            yw1 = max(int(y1 - self.margin * h), 0)
            xw2 = min(int(x2 + self.margin * w), img_w - 1)
            yw2 = min(int(y2 + self.margin * h), img_h - 1)
            face_crop = img_RGB[i][yw1:yw2 + 1, xw1:xw2 + 1, :]
            face_crop_resize = F.interpolate(face_crop.permute(2,0,1)[None,:,:,:], [224,224],  mode='bilinear', align_corners=True) 
            cropped[i] = face_crop_resize[0]
        return cropped

    def normalize_ages(self, age, rmin = 0, rmax = 100, tmin = -1, tmax = 1):
        z = ((age - rmin) / (rmax - rmin)) * (tmax - tmin) + tmin
        return torch.round(z, decimals=4)

class AgeEstimator():
    def __init__(self):
        root = os.path.expanduser('~')
        self.age_model_path = os.path.join(root, "Documents/eg3d-age/eg3d/networks/dex", 'pth/age_sd.pth')
        self.age_model = Age()
        self.age_model.load_state_dict(torch.load(self.age_model_path))
        self.age_model.requires_grad_(requires_grad=False) # weights are freezed 
        self.age_model.eval()
        self.n = torch.arange(0,101)

    def estimate_age(self, gen_img):
        """Takes output of G.synthesis and estimates the age of the synthetic person

        Args:
            gen_img (tensor): image

        Returns:
            tensor: age
        """
        img = (gen_img * 127.5 + 128).clamp(0, 255)
        img = self.resize(img) # resize to fit model
        img = torch.floor(img) # round of pixels
        img = img.type('torch.FloatTensor') # input type of age model
        age_predictions = self.predict_ages(img)
        age_predictions = self.normalize_ages(age_predictions)
        return age_predictions

    def normalize_ages(self, age, rmin = 5, rmax = 80, tmin = -1, tmax = 1):
        z = ((age - rmin) / (rmax - rmin)) * (tmax - tmin) + tmin
        return torch.round(z, decimals=4)


    def normalize(x, rmin = 5, rmax = 80, tmin = -1, tmax = 1):
        z = ((x - rmin) / (rmax - rmin)) * (tmax - tmin) + tmin
        return round(z, 4)

    def predict_ages(self, images):
        P_predictions = self.age_model(images)
        ages = torch.sum(P_predictions * self.n, axis=1)
        return ages

    def resize(self, img):
        """Resize image tensor to fit age model

        Args:
            img (tensor): 

        Returns:
            tensor: resized tensor
        """
        return F.interpolate(img, [224,224],  mode='bilinear', align_corners=True)    

def get_model(model_name="se_resnext50_32x4d", num_classes=101, pretrained="imagenet"):
    model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
    dim_feats = model.last_linear.in_features
    model.last_linear = nn.Linear(dim_feats, num_classes)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    return model
if __name__ == "__main__":
    age = AgeEstimator()