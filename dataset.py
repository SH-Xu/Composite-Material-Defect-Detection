#-----------------------------------------------------------------------#
#                          Library imports                              #
#-----------------------------------------------------------------------#
from transforms import Resize, Rotate, HorizontalFlip, VerticalFlip,\
     Normalize, ToTensor
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
import random
from random import shuffle
import os
import numpy as np
import torch

import torchvision.transforms


class DefectDetectionDataset(Dataset):    
    def __init__(self, images_path_list, set_name, output_size = (320, 480), no_transform =  False):
        super().__init__()
        self.images_path_list = images_path_list
        self.set_name = set_name
        self.output_size = output_size
        self.no_transform = no_transform

    def transform(self, image, mask, set_name, no_transform=False):
        """
        Args:
            image:        image in PIL
            mask:         mask in PIL
            set_name:            Type of partition, either train, valid, or test
            no_transform: False if augmentation is required   
        Returns:
            image and mask pair in torch tensor.
        """
        if (set_name == ('train' or 'val')) and no_transform == False:
            # crop
            rect = torchvision.transforms.RandomResizedCrop.get_params(image, scale=(0.3, 0.8), ratio=(0.5, 1.5))
            image = torchvision.transforms.functional.crop(image, *rect)
            mask = torchvision.transforms.functional.crop(mask, *rect)
            
            t1 = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
            image = t1(image)
            image = image.convert('L')
            mask = mask.convert('L')

            re_t = torchvision.transforms.Resize(self.output_size, torchvision.transforms.InterpolationMode.NEAREST)
            image = re_t(image)
            mask = re_t(mask)

            # Random horizontal flipping with 50% probability
            if random.random() > 0.5:
                t2 = HorizontalFlip()
                image = t2(image)
                mask = t2(mask)

            # Random vertical flipping with 50% probability
            if random.random() > 0.5:
                t3 = VerticalFlip()
                image = t3(image)
                mask = t3(mask)

            # Rotate
            angle = random.choice([0, -90, 90, 180])
            t4 = Rotate(angle)
            image = t4(image)
            mask = t4(mask)
        else:
            image = image.convert('L')
            mask = mask.convert('L')
            re_t = torchvision.transforms.Resize(self.output_size, torchvision.transforms.InterpolationMode.NEAREST)
            image = re_t(image)
            mask = re_t(mask)

        # Transform to tensor
        t5 = ToTensor()
        image = t5(image)
        mask = t5(mask)

        return image, mask

    def __len__(self):
        return len(self.images_path_list)

    def __getitem__(self, idx):
        # Generate one batch of data
        # Open the image file which is in jpg     
        image = Image.open(self.images_path_list[idx])
        # The mask is in png. 
        # Use the image path, and change its extension to png to get the mask's path.
        mask = Image.open(os.path.splitext(self.images_path_list[idx])[0]+'.png') 
        
        # Transform the image and mask PILs to torch tensors. 
        # Perform augmentation if required.
        image, mask = self.transform(image, mask, self.set_name, self.no_transform)
        
        #return the image and mask pair tensors
        return image, mask