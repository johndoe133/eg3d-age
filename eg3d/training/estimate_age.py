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
    """Takes the device as input.
    """
    def __init__(self, device, age_min=0, age_max=100):
        root = os.path.expanduser('~')
        model = get_model(model_name=cfg.MODEL.ARCH, pretrained=None)
        self.age_model = model.to(device)
        pretrained_model_path = os.path.join(root, "Documents/eg3d-age/eg3d/networks/yu4u.pth")
        checkpoint = torch.load(pretrained_model_path)
        self.age_model.load_state_dict(checkpoint['state_dict'])
        self.age_model.requires_grad_(requires_grad=False) 
        self.age_model.eval()
        self.device = device
        self.margin = 0.4
        self.detector = dlib.get_frontal_face_detector()
        self.img_size = 224
        self.ages = torch.arange(0,101, device = self.device)
        self.age_min = age_min
        self.age_max = age_max
    
    def estimate_age(self, gen_img, normalize = True, crop=False):
        """Takes output of G.synthesis and estimates the age of the synthetic person.
        Input shape is [batch_size, channels, w, h]. Images are crop based on the detections.
        The cropped images are converted to BGR and estimated using a pretrained age estimator.

        The method will return the sum of logits in an age range if categories are defined when
        the module is initialized. E.g. if the categories are [0,5,50,101], the first 5 logits are
        summed, the next 45 and the last 50. 

        Args:
            gen_img (tensor): image
            normalized (bool): true if the output age should be normalized. Default is true.
            crop (bool): whether to crop the images before doing age estimation. Default is False. 
        Returns:
            tensor, tensor: predicted ages, logits
        """
        img_RGB = (gen_img * 127.5 + 128).clamp(0, 255) # rescale image values to be between 0 and 255
        if crop:
            img_RGB_detached = img_RGB.detach() 
            detections = self.detect_faces(img_RGB_detached)
            crops = self.crop_images(detections, img_RGB)
        else:
            # resize to fit age model
            crops = F.interpolate(img_RGB, [224,224],  mode='bilinear', align_corners=True) 
        crops = crops.to(self.device)
        crops_BGR = crops.flip([1]) # convert to BGR
        logits = self.age_model(crops_BGR)
        outputs = F.softmax(logits, dim=-1)
        predicted_ages = (outputs * self.ages).sum(axis=-1)

        # if len(self.categories) > 1:
        #     logits = self.categorize_logits(logits) unused

        if normalize:
            return self.normalize_ages(predicted_ages, rmin=self.age_min, rmax=self.age_max), logits
        else:
            return predicted_ages, logits

    def categorize_logits(self, logits):
        """Will categorize the logits into the age ranges specified by the categories list
        by summing the logits in each category.
        Only relevant if `len(self.categories) > 1`, otherwise categories are unused. 

        Args:
            logits (tensor): raw output of the age model with the size [batch_size, 101]

        Returns:
            tensor: logits that are summed in the given categories.
        """
        batch, n = logits.shape
        zeros = torch.zeros((batch, len(self.categories) - 1), device = self.device)
        buckets = torch.bucketize(self.ages, torch.tensor(self.categories, device = self.device), right=True)
        logits_categorized = zeros.index_add_(1, buckets - 1, logits)
        return logits_categorized

    def estimate_age_rgb(self, image, normalize = True, crop = False):
        image = image.type('torch.FloatTensor')
        if crop:
            detections = self.detect_faces(image)
            image = image.permute(0,3,1,2)
            crops = self.crop_images(detections, image)
        else:
            crops = F.interpolate(image.permute(0,3,1,2), [224,224],  mode='bilinear', align_corners=True) 
        crops = crops.to(self.device)
        crops_BGR = crops.flip([1])
        outputs = F.softmax(self.age_model(crops_BGR), dim=-1)
        predicted_ages = (outputs * self.ages).sum(axis=-1)
        if normalize:
            return self.normalize_ages(predicted_ages)
        else:
            return predicted_ages

    def estimate_age_testing(self, image):
        outputs = F.softmax(self.age_model(image), dim=-1)
        predicted_ages = (outputs * self.ages).sum(axis=-1)
        return predicted_ages

    def detect_faces(self, img_RGB):
        """Detects faces using `dlib.get_frontal_face_detector()`. Return a list
        of coordinates used to crop the image. Images are resized to 640 x 640 before
        detection.

        Args:
            img_RGB (tensor): input shape [batch_size, channels, w, h]

        Returns:
            list: detections
        """
        img_RGB = img_RGB.permute(0,3,1,2)
        
        detections = [] # x1, y1, x2, y2, w, h 
        img_RGB = F.interpolate(img_RGB, [640, 640],  mode='bilinear', align_corners=False) 
        
        img_RGB = img_RGB.permute(0,2,3,1) # converted to shape [batch_size, w, h, channels] to fit dlib detector
        
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
        """Crops the images using predetermined coordinates from `detect_faces`.
        Resizes the cropped image to fit the age model.

        Args:
            detections (list): list of detections
            img_RGB (tensor):  input shape [batch_size, channels, w, h]

        Returns:
            tensor: tensor of images cropped and resized to fit age model
        """
        batch_size, channels , img_w, img_h = img_RGB.shape[0] , img_RGB.shape[1], img_RGB.shape[2], img_RGB.shape[3]
        cropped = torch.empty(batch_size, channels, self.img_size, self.img_size)
        for i in range(batch_size):
            
            x1, y1, x2, y2, w, h = detections[i]
            xw1 = max(int(x1 - self.margin * w), 0)
            yw1 = max(int(y1 - self.margin * h), 0)
            xw2 = min(int(x2 + self.margin * w), img_w - 1)
            yw2 = min(int(y2 + self.margin * h), img_h - 1)
            face_crop = img_RGB[i][:, yw1:yw2 + 1, xw1:xw2 + 1]
            face_crop_resize = F.interpolate(face_crop[None,:,:,:], [224,224],  mode='bilinear', align_corners=False) 
            cropped[i] = face_crop_resize[0]
        return cropped

    def normalize_ages(self, age, rmin = 0, rmax = 100, tmin = -1, tmax = 1):
        z = ((age - rmin) / (rmax - rmin)) * (tmax - tmin) + tmin
        return torch.round(z, decimals=4)

class AgeEstimator():
    def __init__(self, age_min=0, age_max=100):
        root = os.path.expanduser('~')
        self.age_model_path = os.path.join(root, "Documents/eg3d-age/eg3d/networks/dex", 'pth/age_sd.pth')
        self.age_model = Age()
        self.age_model.load_state_dict(torch.load(self.age_model_path))
        self.age_model.requires_grad_(requires_grad=False) # weights are freezed 
        self.age_model.eval()
        self.n = torch.arange(0,101)
        self.age_min = age_min
        self.age_max = age_max

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
        age_predictions, logits = self.predict_ages(img)
        age_predictions = self.normalize_ages(age_predictions, rmin=self.age_min, rmax=self.age_max)
        return age_predictions, logits

    def normalize_ages(self, age, rmin = 5, rmax = 80, tmin = -1, tmax = 1):
        z = ((age - rmin) / (rmax - rmin)) * (tmax - tmin) + tmin
        return torch.round(z, decimals=4)


    def normalize(x, rmin = 5, rmax = 80, tmin = -1, tmax = 1):
        z = ((x - rmin) / (rmax - rmin)) * (tmax - tmin) + tmin
        return round(z, 4)

    def predict_ages(self, images):
        P_predictions = self.age_model(images)
        ages = torch.sum(P_predictions * self.n, axis=1)
        return ages, P_predictions

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
    age_model = AgeEstimatorNew(torch.device("cuda"))
    p = "/zhome/d7/6/127158/Documents/eg3d-age/age-estimation-pytorch/in"
    
    for path in sorted(os.listdir(os.path.join(p))):
        if 'dataset' in path:
            continue
        image_path = os.path.join(p, path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(image)
        img_tensor = img_tensor.to("cuda:0")
        predicted_age = age_model.estimate_age_rgb(img_tensor[None,:,:,:], crop=True, normalize=False).item()
        print(path, "age:",predicted_age)
        