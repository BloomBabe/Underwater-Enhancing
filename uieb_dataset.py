import numpy as np 
import math
import os
import json
from torch.utils.data import Dataset
from PIL import Image

class UiebDataset(Dataset):

    def __init__(self,
                 raw_pth,
                 reference_pth):
        super(UiebDataset, self).__init__()
        self.raw_pth = raw_pth
        self.reference_pth = reference_pth

        self.raw_filenames = sorted(os.listdir(self.raw_pth))
        self.reference_filenames = sorted(os.listdir(self.reference_pth))

        assert len(self.raw_filenames) == len(self.reference_filenames)
    
    def __len__(self):
        return 2*len(self.raw_filenames)

    def __getitem__(self, idx):
        raw_file = self.raw_filenames[idx]
        if raw_file not in reference_filenames:
            raise ValueError(f'{raw_file} does not exist in {reference_pth}')
        raw_img = np.asarray(PIL.Image.open(os.path.join(self.raw_pth, self.raw_file)))
        ref_img = np.asarray(PIL.Image.open(os.path.join(self.reference_pth, self.raw_file)))
        return raw_img, ref_img