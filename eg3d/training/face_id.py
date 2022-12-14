import os
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import cv2
import sys
sys.path.append("/zhome/d7/6/127158/Documents/eg3d-age/eg3d")
from networks.MagFace.network_inf import builder_inf
from torchvision.models import resnet50
import torchvision
from networks.MagFace.iresnet import iresnet100, iresnet50
from training.mtcnn import MTCNN
from skimage import transform as trans
from kornia.geometry import warp_affine
try:
    import insightface 
except:
    print("Didnt import insightface")
from easydict import EasyDict
import onnxruntime as ort
ort.set_default_logger_severity(3)

class FaceIDLoss:
    """Module used for identity preservation. The available models for training are FaceNet and MagFace.
    ArcFace can only be used to evaluate since it isnt implemented in pytorch in this class.
    
    """
    def __init__(self, device, model = "FaceNet", resize_img = True):
        self.model = model
        self.mtcnn = MTCNN(device=device, selection_method="probability", thresholds=[0.2, 0.2, 0.2])
        self.device = device
        
        if self.model == "FaceNet":
            self.align = self.mtcnn
            self.id_model = InceptionResnetV1(pretrained='vggface2', device=device).requires_grad_(requires_grad=False).eval()
            self.resize_shape = 160 # resize to 160 x 160
        
        elif self.model == "MagFace":
            # bruger https://github.com/deepinsight/insightface/blob/cdc3d4ed5de14712378f3d5a14249661e54a03ec/python-package/insightface/utils/face_align.py
            # til alignment 
            self.align = self.MagFaceAlign 
            class args_new:
                arch = 'iresnet100'
                embedding_size = 512
                resume = "./networks/MagFace/iResNet100_MagFace.pth"
                cpu_mode = None
            self.id_model = builder_inf(args_new())
            self.id_model = self.id_model.to(device).requires_grad_(requires_grad=False).eval()
            self.resize_shape = 112 # resize to 112 x 112
        
        elif self.model == "ArcFace":
            self.detection_model = self.load_face_detection_model('./networks/det_10g.onnx')
            self.fr_model = self.load_face_recognition_model('./networks/ms1mv3_r100.onnx')
            self.align = self.ArcFaceAlign # to be done
            # https://onedrive.live.com/?authkey=%21AFZjr283nwZHqbA&cid=4A83B6B633B029CC&id=4A83B6B633B029CC%215751&parId=4A83B6B633B029CC%215585&o=OneUp
            # self.id_model = iresnet50()
            # self.id_model.load_state_dict(torch.load("./networks/arcfacers50.pth"))
            # self.id_model = self.id_model.to(device).requires_grad_(requires_grad=False).eval()
            self.resize_shape = 112 # resize to 112 x 112

        self.resize_img = resize_img

    def get_feature_vector_arcface(self, imgs):
        # only for evaluation
        if self.model != "ArcFace":
            raise AttributeError
        if torch.is_tensor(imgs):
            imgs = (imgs.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            imgs = imgs[0].cpu().numpy()
        detected_face = self.detection_model.detect(imgs)
        detected_face = EasyDict({'kps': detected_face[1][0], 'det':detected_face[0][0]})
        
        face_embedding = self.fr_model.get(imgs, detected_face)   
        return torch.tensor(face_embedding[None,:]) # return type so it can be used in torch.cosineSimilarity()


    def load_face_detection_model(self, det_model_path):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        det_model = insightface.model_zoo.get_model(det_model_path, providers=providers)
        det_model.prepare(ctx_id=0, det_size=(512, 512), det_thresh=0.01, input_size = (512, 512))
        return det_model

    def load_face_recognition_model(self, fr_model_path):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        fr_model = insightface.model_zoo.get_model(fr_model_path, providers=providers)
        fr_model.prepare(ctx_id =0)
        return fr_model

    def get_feature_vector(self, img):
        """Extracts feature vector using `self.id_model`. Will align according to what
        the authors of `self.id_model` did. If no face is detected for one or several images
        in the batch, the image is resized to the correct image size and the feature vector is 
        extracted without the alignment. 

        Args:
            img (tensor): batch of images generated by G

        Returns:
            tensor: the extracted feature vector using `self.id_model` for each image in the batch
        """
        img_RGB = self.transform_to_RGB(img) 
        aligned = self.align(img_RGB.permute(0,2,3,1)) # shape is converted to [batch, w, h, channels]
        if None in aligned: # a face is not detected 
            aligned_updated = []
            for i, aligned_image in enumerate(aligned):
                if aligned_image is None:
                    only_resized = self.resize(img[i][None,:,:,:])
                    only_resized_reshaped = only_resized[0].permute(1,2,0)
                    aligned_updated.append(only_resized_reshaped)
                else:
                    aligned_updated.append(aligned[i])
            aligned = aligned_updated
    
        aligned = torch.stack(aligned)

        feature_vector = self.id_model(aligned.permute(0,3,1,2)) # id_model takes shape [batch, channels, w, h]
        return feature_vector

    def MagFaceAlign(self, imgs):
        """Standard alignment process of MagFace. 

        Args:
            imgs (tensor): image tensor

        Returns:
            list: aligned images, None if face was not detected 
        """
        #imgs_rgb = self.transform_to_RGB(imgs)
        aligned_imgs = []
        imgs_rgb = imgs # delete
        bboxes, probs, landmarks = self.mtcnn.detect(imgs_rgb, landmarks=True) 
        
        for i, (bbox, prob, landmark) in enumerate(zip(bboxes, probs, landmarks)):
            if prob is None: # no face detected
                aligned_imgs.append(None) # handled in get_feature_vector
            else: # face detected
                source = np.array([
                    [30.2946, 51.6963],
                    [65.5318, 51.5014],
                    [48.0252, 71.7366],
                    [33.5493, 92.3655],
                    [62.7299, 92.2041]
                ], dtype=np.float32)
                source[:,0] += 8.0
                destination = landmark[0].astype(np.float32) # to match type between source and destination
                tform = trans.SimilarityTransform()
                tform.estimate(destination, source)
                M = tform.params[0:2,:]
                M = torch.Tensor(M).to(self.device)
                if M is None:
                    print("Im in trouble")
                else:
                    # it's warping time
                    aligned_img = warp_affine(imgs_rgb[i].permute(2,0,1)[None,:,:,:], M[None,:,:], dsize=(112,112))[0]
                    aligned_imgs.append(aligned_img.permute(1,2,0)) # reshape to [w, h, channels] to fit mtcnn output
        
        return aligned_imgs

    def ArcFaceAlign(self, img):
        raise NotImplementedError

    def get_feature_vector_test(self, img):
        """To test the function of extracting feature vectors. Usage is inputting a tensor from a image
        loaded using cv2:

        ```
        image1 = cv2.imread(p1)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(image1)
        img_tensor = img_tensor.to(device)
        v = model.get_feature_vector_test(img_tensor[None,:,:,:].float())
        ```

        Args:
            img (tensor): 

        Returns:
            tensor: feature vector
        """
        aligned = self.align(img)
        aligned = torch.stack(aligned)

        feature_vector = self.id_model(aligned.permute(0,3,1,2)) # id_model takes shape [batch, channels, w, h]
        return feature_vector
        # if aligned is None:
        #     only_resized = self.resize(img.permute(2,0,1)[None,:,:,:])
        #     feature_vector = self.id_model(only_resized)
        # else:
        #     feature_vector = self.id_model(aligned.permute(2,0,1)[None,:,:,:])
        # return feature_vector

    def transform_to_RGB(self, img):
        img255 = (img * 127.5 + 128).clamp(0, 255)

        return img255.float()

    def resize(self, img):
        """Resize image tensor to fit age model

        Args:
            img (tensor): 

        Returns:
            tensor: resized tensor
        """
        return F.interpolate(img, [self.resize_shape, self.resize_shape],  mode='bilinear', align_corners=True)    
    
    def __repr__(self) -> str:
        return f"{self.id_model} \n\n ID_model using {self.model}"

    def __str__(self) -> str:
        return f"{self.id_model} \n\n ID_model using {self.model}"

# MIT License

# Copyright (c) 2019 Timothy Esler
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software 
# and associated documentation files (the "Software"), to deal in the Software without restriction, 
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, 
# subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial 
# portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT 
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# https://github.com/timesler/facenet-pytorch/blob/master/models/inception_resnet_v1.py
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False
        ) # verify bias false
        self.bn = nn.BatchNorm2d(
            out_planes,
            eps=0.001, # value found in tensorflow
            momentum=0.1, # default pytorch value
            affine=True
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Block35(nn.Module):

    def __init__(self, scale=1.0):
        super().__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(256, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.conv2d = nn.Conv2d(96, 256, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Block17(nn.Module):

    def __init__(self, scale=1.0):
        super().__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(896, 128, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(896, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 128, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(128, 128, kernel_size=(7,1), stride=1, padding=(3,0))
        )

        self.conv2d = nn.Conv2d(256, 896, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Block8(nn.Module):

    def __init__(self, scale=1.0, noReLU=False):
        super().__init__()

        self.scale = scale
        self.noReLU = noReLU

        self.branch0 = BasicConv2d(1792, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1792, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=(1,3), stride=1, padding=(0,1)),
            BasicConv2d(192, 192, kernel_size=(3,1), stride=1, padding=(1,0))
        )

        self.conv2d = nn.Conv2d(384, 1792, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class Mixed_6a(nn.Module):

    def __init__(self):
        super().__init__()

        self.branch0 = BasicConv2d(256, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(256, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1),
            BasicConv2d(192, 256, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Mixed_7a(nn.Module):

    def __init__(self):
        super().__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionResnetV1(nn.Module):
    """Inception Resnet V1 model with optional loading of pretrained weights.
    Model parameters can be loaded based on pretraining on the VGGFace2 or CASIA-Webface
    datasets. Pretrained state_dicts are automatically downloaded on model instantiation if
    requested and cached in the torch cache. Subsequent instantiations use the cache rather than
    redownloading.
    Keyword Arguments:
        pretrained {str} -- Optional pretraining dataset. Either 'vggface2' or 'casia-webface'.
            (default: {None})
        classify {bool} -- Whether the model should output classification probabilities or feature
            embeddings. (default: {False})
        num_classes {int} -- Number of output classes. If 'pretrained' is set and num_classes not
            equal to that used for the pretrained model, the final linear layer will be randomly
            initialized. (default: {None})
        dropout_prob {float} -- Dropout probability. (default: {0.6})
    """
    def __init__(self, pretrained=None, classify=False, num_classes=None, dropout_prob=0.6, device=None):
        super().__init__()

        # Set simple attributes
        self.pretrained = pretrained
        self.classify = classify
        self.num_classes = num_classes

        if pretrained == 'vggface2':
            tmp_classes = 8631
        elif pretrained == 'casia-webface':
            tmp_classes = 10575
        elif pretrained is None and self.classify and self.num_classes is None:
            raise Exception('If "pretrained" is not specified and "classify" is True, "num_classes" must be specified')


        # Define layers
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.conv2d_4b = BasicConv2d(192, 256, kernel_size=3, stride=2)
        self.repeat_1 = nn.Sequential(
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
        )
        self.mixed_6a = Mixed_6a()
        self.repeat_2 = nn.Sequential(
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
        )
        self.mixed_7a = Mixed_7a()
        self.repeat_3 = nn.Sequential(
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
        )
        self.block8 = Block8(noReLU=True)
        self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_prob)
        self.last_linear = nn.Linear(1792, 512, bias=False)
        self.last_bn = nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)

        if pretrained is not None:
            self.logits = nn.Linear(512, tmp_classes)
            load_weights(self, pretrained)

        if self.classify and self.num_classes is not None:
            self.logits = nn.Linear(512, self.num_classes)

        self.device = torch.device('cpu')
        if device is not None:
            self.device = device
            self.to(device)

    def forward(self, x):
        """Calculate embeddings or logits given a batch of input image tensors.
        Arguments:
            x {torch.tensor} -- Batch of image tensors representing faces.
        Returns:
            torch.tensor -- Batch of embedding vectors or multinomial logits.
        """
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        x = self.avgpool_1a(x)
        x = self.dropout(x)
        x = self.last_linear(x.view(x.shape[0], -1))
        x = self.last_bn(x)
        if self.classify:
            x = self.logits(x)
        else:
            x = F.normalize(x, p=2, dim=1)
        return x


def load_weights(mdl, name):
    """Download pretrained state_dict and load into model.
    Arguments:
        mdl {torch.nn.Module} -- Pytorch model.
        name {str} -- Name of dataset that was used to generate pretrained state_dict.
    Raises:
        ValueError: If 'pretrained' not equal to 'vggface2' or 'casia-webface'.
    """
    if name == 'vggface2':
        path = 'https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt'
    elif name == 'casia-webface':
        path = 'https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180408-102900-casia-webface.pt'
    else:
        raise ValueError('Pretrained models only exist for "vggface2" and "casia-webface"')

    model_dir = r"./networks/face_id_model.pt" # from link https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt

    if not os.path.exists(model_dir):
        print('path doesn\'t exist')

    state_dict = torch.load(model_dir)
    mdl.load_state_dict(state_dict)


def get_torch_home():
    torch_home = os.path.expanduser(
        os.getenv(
            'TORCH_HOME',
            os.path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')
        )
    )
    return torch_home


if __name__=="__main__":
    device = torch.device("cuda")
    arcface = FaceIDLoss(device, model="ArcFace")
    magface = FaceIDLoss(device, model="MagFace")
    # p = '/zhome/d7/6/127158/Documents/eg3d-age/age-estimation-pytorch/in/img00000022.png'
    # p2 = '/zhome/d7/6/127158/Documents/eg3d-age/age-estimation-pytorch/in/img00000024.png'
    # image1 = cv2.imread(p)
    # # image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    # img_tensor = torch.from_numpy(image1).float()
    # img_tensor = img_tensor.to(device).requires_grad_(True)
    
    # f1 = model.get_feature_vector_arcface(img_tensor[None,:,:,:].permute(0,3,1,2))
    # print(f1)
    # image2 = cv2.imread(p2)
    # # image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    # img_tensor2 = torch.from_numpy(image2)
    # img_tensor2 = img_tensor2.to(device)
    # img_3 = torch.ones((3,512,512)).to(device)
    # img = torch.stack([img_3, img_tensor.permute(2,0,1), img_tensor2.permute(2,0,1)])

    # f2 = model.get_feature_vector_arcface(image2)

    # # stack image1 og 2 og se om det virker med alignment
    # cosine_sim_f = torch.nn.CosineSimilarity()
    # print("SIM:", cosine_sim_f(torch.tensor([f1]), torch.tensor([f2])))
    # # print("Norm", np.linalg.norm(v2.detach().cpu().numpy()))
    from itertools import tee
    def pairwise(iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)
    appa = r"/work3/morbj/APPA/appa-real-release/valid"
    cosine_sim_f = torch.nn.CosineSimilarity()
    ffhq="/work3/morbj/FFHQ/00000"
    image_names = list(filter(lambda x: False if ('.mat' in x or '_face' in x) else True, os.listdir(ffhq)))
    i=0
    sims=[]
    sims_mag = []
    for img1, img2 in pairwise(image_names):
        if i > 100:
            break

        p1 = os.path.join(ffhq, img1)
        p2 = os.path.join(ffhq, img2)

        image1 = cv2.imread(p1)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(image1)
        img_tensor = img_tensor.to(device)

        image1 = cv2.imread(p1)
        image2 = cv2.imread(p2)

        f1 = arcface.get_feature_vector_arcface(image1)
        f2 = arcface.get_feature_vector_arcface(image2)

        image_tensor1 = torch.from_numpy(image1).to(device)
        image_tensor2 = torch.from_numpy(image2).to(device)
        f1_mag = magface.get_feature_vector_test(image_tensor1[None,:,:,:].float())
        f2_mag = magface.get_feature_vector_test(image_tensor2[None,:,:,:].float())
    
        sim = cosine_sim_f(torch.tensor(f1), torch.tensor(f2)).item()
        sims.append(sim)

        sim_mag = cosine_sim_f(torch.tensor(f1_mag), torch.tensor(f2_mag)).item()
        sims_mag.append(sim_mag)
        print(f"{p1}, {p2} SIM:", sim)
        i+= 1
        
    sims=np.array(sims)
    sims_mag=np.array(sims_mag)
    print("ArcFace Average non-mated similarity score", sims.mean())
    print("MagFace Average non-mated similarity score", sims_mag.mean())

