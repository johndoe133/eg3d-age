import cv2
import numpy as np

class AgeEstimator():
    """
    Class to estimate ages of images.
    """   
    def __init__(self, device = "cpu", age_model_input_shape = (224, 224)):
        self.age_model_input_shape = age_model_input_shape # input shape for age model
        self.device = device
        self.output_indexes = np.array([i for i in range(0, 101)])
        self.config_file = r"./networks/age_model/age.prototxt"
        self.model_file = r"./networks/age_model/dex_chalearn_iccv2015.caffemodel"
        self.agenet = cv2.dnn.readNetFromCaffe(self.config_file, self.model_file)
        if self.device == "cpu":
            self.agenet.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
        if self.device == "gpu":
            self.agenet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.agenet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        
    def resize_img(self, images):
        """Resizes array of images to fit input size of age model.

        Args:
            images np.array: images should be BGR (default opencv format) but 
            doesn't appear to make a difference if not.

        Returns:
            np.array: resized images
        """
        return np.array([cv2.resize(img, self.age_model_input_shape) for img in images])
    
    def convert_images(self, images):
        """Converts normal distributed tensor 3x512x512 to a 0-255 512x512x3 numpy array

        Args:
            images torch.tensor: Images as given by model

        Returns:
            np.array: reorganized images
        """
        images = images.permute(0,2,3,1) # from img * channel * x * y to img * x * y * channels
        images_numpy = images.cpu().detach().numpy()
        return images_numpy

    def estimate(self, images):
        """Estimates age of array of images

        Args:
            images np.array: resized images

        Returns:
            np.array: predicted ages
        """
        images_converted = self.convert_images(images)
        images_resized = self.resize_img(images_converted)
        img_blob = cv2.dnn.blobFromImages(images_resized)
        self.agenet.setInput(img_blob)
        age_dist = self.agenet.forward()
        prediction = np.round(np.sum(age_dist * self.output_indexes, axis=1), 2)
        return prediction