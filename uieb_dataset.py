import numpy as np 
import math
import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image

class UiebDataset(Dataset):

    def __init__(self,
                 raw_pth,
                 reference_pth,
                 image_size = (256, 256)):
        super(UiebDataset, self).__init__()
        self.image_size = image_size
        self.raw_pth = raw_pth
        self.reference_pth = reference_pth

        self.raw_filenames = sorted(os.listdir(self.raw_pth))
        self.reference_filenames = sorted(os.listdir(self.reference_pth))

        assert len(self.raw_filenames) == len(self.reference_filenames)

    def __len__(self):
        return len(self.raw_filenames)

    def _load_img(self, pth):
        img = Image.open(pth)
        img = img.resize(self.image_size, Image.ANTIALIAS)
        out = np.asarray(img) / 255.
        # img = ToTensor()(img)
        # img = torch.unsqueeze(img, 0)
        # print(f'tensor size: {img.size()}')
        # out = F.interpolate(img)
        # print(f'interpolate tensor size: {out.size()}')
        # out = out.numpy()/255.
        return out

    def __getitem__(self, idx):
        raw_file = self.raw_filenames[idx]
        if raw_file not in self.reference_filenames:
            raise ValueError(f'{raw_file} does not exist in {reference_pth}')
        raw_img = self._load_img(os.path.join(self.raw_pth, raw_file))
        ref_img = self._load_img(os.path.join(self.reference_pth, raw_file))
                        #resize(self.image_size, Image.ANTIALIAS)
        #print(raw_img.shape)
        return raw_img, ref_img