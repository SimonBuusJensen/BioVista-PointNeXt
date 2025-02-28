import os
import sys
from PIL import Image
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torch.nn as nn



def set_val_data_transforms(img_sz=240):
    transforms = T.Compose([
        T.Resize((img_sz, img_sz)), # Resize to ResNet input size
    ])
    return transforms

def set_train_data_transforms(img_sz=240):
    transform_list = [
        T.Resize((img_sz, img_sz))  # Resize to the desired input size
    ]
    transform_list.append(T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5))

    transforms = T.Compose(transform_list)      
    return transforms

def load_image(image_fp):
    img = Image.open(image_fp)
    img = img.convert("RGB")
    return img
