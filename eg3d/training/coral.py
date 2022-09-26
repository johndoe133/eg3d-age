# coding: utf-8

#############################################
# Consistent Cumulative Logits with ResNet-34

# https://github.com/Raschka-research-group/coral-cnn
#############################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

from torchvision import transforms
from PIL import Image

class Coral:
    """Class to contain the Coral face age estimation module.
    Uses a pretrained model, trained on the MORPH-2 dataset, which
    has ages from 16-70. It can, therefore, predict in this range.
    Args:
            dataset (str, optional): Dataset pretrained model was trained on. 
            Defaults to "morph2".
            model_name (str, optional): Name of the pretrained model. Defaults to "coral_morph".
            GRAYSCALE (bool, optional): Whether the image is gray scale. Defaults to False.
            verbose (bool, optional): Whether to print the output of estimating ages. Defaults to False. 
    """
    def __init__(self, dataset="morph2", model_name="coral_morph", GRAYSCALE=False, verbose=False):
        torch.backends.cudnn.deterministic = True
        self.dataset = dataset
        self.verbose = verbose
        self.tensor_to_PIL = transforms.ToPILImage()
        if self.dataset == 'afad':
            self.NUM_CLASSES = 26
            self.ADD_CLASS = 15
        elif self.dataset == 'morph2':
            self.NUM_CLASSES = 55
            self.ADD_CLASS = 16
        elif self.dataset == 'cacd':
            self.NUM_CLASSES = 49
            self.ADD_CLASS = 14
        else:
            raise ValueError("args.dataset must be 'afad',"
                            " 'morph2', or 'cacd'. Got %s " % (dataset))
        self.device = torch.device('cuda')
        STATE_DICT_PATH = f'./networks/{model_name}.pt'
        self.model = resnet34(self.NUM_CLASSES, GRAYSCALE, self.device)
        self.model.load_state_dict(torch.load(STATE_DICT_PATH, map_location=self.device))
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()

    def estimate_age(self, image):
        """
        Args:
            image_path (str): Path to the image we want estimated

        Returns:
            int: estimated age
        """  
        image = self.load_image(image)
        image = image.unsqueeze(0)

        with torch.set_grad_enabled(False):
            logits, probas = self.model(image)
            predict_levels = probas > 0.5
            predicted_label = torch.sum(predict_levels, dim=1)
            predicted_age = predicted_label + self.ADD_CLASS
            if self.verbose:
                print('Class probabilities:', probas)
                print('Predicted class label:', predicted_label.item())
                print('Predicted age in years:', predicted_age.item())
        return predicted_age

    def load_image(self, image):
        if type(image) is str:
            image = Image.open(image)
        elif torch.is_tensor(image):
            image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255)
            image = self.tensor_to_PIL(image[0].cpu())
        custom_transform = transforms.Compose([transforms.Resize((128, 128)),
                                            transforms.CenterCrop((120, 120)),
                                            transforms.ToTensor()])
        image = custom_transform(image)
        image = image.to(self.device)
        return image



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale, device):
        self.num_classes = num_classes
        self.inplanes = 64
        self.device = device
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4)
        self.fc = nn.Linear(512, 1, bias=False)
        self.linear_1_bias = nn.Parameter(torch.zeros(self.num_classes-1).to(self.device).float())

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        logits = logits + self.linear_1_bias
        probas = torch.sigmoid(logits)
        return logits, probas


def resnet34(num_classes, grayscale, device):
    """Constructs a ResNet-34 model."""
    model = ResNet(block=BasicBlock,
                   layers=[3, 4, 6, 3],
                   num_classes=num_classes,
                   grayscale=grayscale,
                   device=device)
    return model

if __name__ == '__main__':
    dataset = 'morph2'
    image_path = '/zhome/d7/6/127158/Documents/eg3d-age/eg3d/datasets/FFHQ_512_6_balanced/00003/img00003760.png'
    coral = Coral(dataset, verbose=True)
    coral.estimate_age(image_path)

