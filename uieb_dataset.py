import numpy as np 
import math
import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage
from skimage import io, transform

class UiebDataset(Dataset):

    def __init__(self,
                 dataset_pth,
                 mode = 'train',
                 transform = None):
        super(UiebDataset, self).__init__()
        
        self.transform = transform
        self.dataset_pth = dataset_pth
        assert mode in ['train', 'val']
        self.dataset_pth = os.path.join(self.dataset_pth, mode)
        self.raw_pth = os.path.join(self.dataset_pth, 'raw')
        self.raw_filenames = sorted(os.listdir(self.raw_pth))
        self.reference_pth = os.path.join(self.dataset_pth, 'ref')
        self.reference_filenames = sorted(os.listdir(self.reference_pth))

        assert len(self.raw_filenames) == len(self.reference_filenames)

    def __len__(self):
        return len(self.raw_filenames)

    def __getitem__(self, idx):
        raw_file = self.raw_filenames[idx]
        if raw_file not in self.reference_filenames:
            raise ValueError(f'{raw_file} does not exist in {reference_pth}')
        raw_img = io.imread(os.path.join(self.raw_pth, raw_file))
        ref_img = io.imread(os.path.join(self.reference_pth, raw_file))
        sample = {'raw_image': raw_img, 'ref_image': ref_img}                
        
        if self.transform:
            sample = self.transform(sample)
        return sample