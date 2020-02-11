import torch
import torchvision 
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.dataset import random_split
from PIL import Image



### Flux Dataset Dataloading -------------------------------------------------------------------------------
class FluxData(Dataset):
    def __init__(self, root_dir, transform = None):
        self.image_path = root_dir + '/images/'
        self.masks_path = root_dir + '/masks/'
        self.transform  = transform
        self.to_tensor = transforms.ToTensor()
        self.images = sorted(glob.glob(self.image_path+'*.jpg' ))
        self.masks = sorted(glob.glob(self.masks_path+'*.png'))
    def __getitem__(self, idx):
        height = 512
        width = 512
        image = Image.open(self.images[idx])
        image = image.resize((height, width))
        image = np.array(image)
        mask = Image.open(self.masks[idx])
        mask = mask.resize((height, width))
        mask = np.array(mask)
        class_id = [0, 3, 4, 5,37]
        for i in np.unique(mask):
            if i in class_id:
                mask[mask == i] = class_id.index(i)
            else:
                mask[mask == i] = 0
        if self.transform is not None:
            transformed = self.transform(image = image, mask = mask)
            image = transformed['image']
            mask = transformed['mask']
        seg_labels = torch.from_numpy(mask).long()
        image = torch.from_numpy(image).float().permute([2,0,1])
        return image, seg_labels
    def __len__(self):
        return len(self.images)


### BDD Dataset Dataloading -------------------------------------------------------------------------------
class BDD_Data(Dataset):
    def __init__(self, root_dir, transform = None):
        self.image_path = root_dir + '/images/train/'
        self.masks_path = root_dir + '/labels/train/'
        self.transform  = transform
        self.to_tensor = transforms.ToTensor()
        self.images = sorted(glob.glob(self.image_path+'*.jpg' ))
        self.masks = sorted(glob.glob(self.masks_path+'*.png'))
    def __getitem__(self, idx):
        height = 512
        width = 512
        image = cv2.imread(self.images[idx])
        image = cv2.resize(image, (width, height))
        mask = cv2.imread(self.masks[idx], 0)
        mask = cv2.resize(mask, (width, height))
        mask[mask == 0] = 18
        mask[mask == 255] = 0
        class_id = [0, 13, 14, 17,18]
        for i in np.unique(mask):
            if i in class_id:
              mask[mask == i] = class_id.index(i)
            else:
                mask[mask == i] = 0
        seg_labels = torch.from_numpy(mask).long()
        image = torch.from_numpy(image).float().permute([2,0,1])
        return image, seg_labels
    def __len__(self):
        return len(self.images)


class Mapillary(Dataset):
    def __init__(self, root_dir, transform = None):
        self.image_path = root_dir + '/images/'
        self.masks_path = root_dir + '/labels/'
        self.transform  = transform
        self.to_tensor = transforms.ToTensor()
        self.images = sorted(glob.glob(self.image_path+'*.jpg' ))
        self.masks = sorted(glob.glob(self.masks_path+'*.png'))
    def __getitem__(self, idx):
        height = 512
        width = 512
        image = Image.open(self.images[idx])
        image = image.resize((height, width))
        image = np.array(image)
        mask = Image.open(self.masks[idx])
        mask = mask.resize((height, width))
        mask = np.array(mask)
        class_id = [0,19, 55, 61,13] ### unlabeled, person, car, truck, road
        for i in np.unique(mask):
            if i in class_id:
                mask[mask == i] = class_id.index(i)
            else:
                mask[mask == i] = 0
        if self.transform is not None:
            transformed = self.transform(image = image, mask = mask)
            image = transformed['image']
            mask = transformed['mask']
        seg_labels = torch.from_numpy(mask).long()
        image = torch.from_numpy(image).float().permute([2,0,1])
        return image, seg_labels
    def __len__(self):
        return len(self.images)