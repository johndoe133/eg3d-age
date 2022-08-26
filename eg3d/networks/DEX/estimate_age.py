import os
import torch
import numpy as np
import torch.nn.functional as F
from .dex.models import Age, Gender

class AgeEstimator2():
    def __init__(self):
        self.age_model_path = os.path.join("./networks/DEX/dex", 'pth/age_sd.pth')
        self.age_model = Age()
        self.age_model.load_state_dict(torch.load(self.age_model_path))
        self.age_model.eval()
        self.n = torch.arange(0,101)

    def estimate_age(self, gen_img):
        gen_img = self.resize(gen_img) # resize to fit model
        lo, hi = torch.min(gen_img).item(), torch.max(gen_img).item() 
        #adjust pixel values to 0 - 255
        gen_img = (gen_img - lo) * (255 / (hi - lo))
        gen_img = gen_img.type(torch.uint8) # round of pixels
        gen_img = gen_img.type('torch.FloatTensor') # input type of age model
        age_predictions = self.predict_ages(gen_img)
        return age_predictions

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