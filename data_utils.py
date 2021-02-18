import numpy as np
import torch
from skimage import io, transform
import torch.nn.functional as F
from torchvision import transforms

torch.manual_seed(17)

class Resize(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def _resize(self, image):
        h, w = image.size()[1:3]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = F.interpolate(image.unsqueeze(0), (new_h, new_w))
        return img.squeeze(0)

    def __call__(self, sample):
        raw_image, ref_image = sample['raw_image'], sample['ref_image']

        new_raw_image = self._resize(raw_image)
        new_ref_image = self._resize(ref_image)    
        return {'raw_image': new_raw_image, 'ref_image': new_ref_image}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def _transpose(self, image, channels=(2, 0, 1)):
        return image.transpose(channels)

    def __call__(self, sample):
        raw_image, ref_image = sample['raw_image'], sample['ref_image']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        new_raw_image = self._transpose(raw_image)
        new_ref_image = self._transpose(ref_image)

        return {'raw_image': torch.from_numpy(new_raw_image).float(),
                'ref_image': torch.from_numpy(new_ref_image).float()}

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def _normalize(self, image):
        return transforms.Normalize(self.mean, self.std)(image)

    def __call__(self, sample):
        raw_image, ref_image = sample['raw_image'], sample['ref_image']
        norm_raw_image = self._normalize(raw_image)
        norm_ref_image = self._normalize(ref_image)
        return {'raw_image': norm_raw_image,
                'ref_image': norm_ref_image}

class RandomRotation(object):
    """Rotate the image by angle."""
    def _random_rotate(self, image):
        return transforms.RandomRotation()(image)

    def __call__(self, sample):
        raw_image, ref_image = sample['raw_image'], sample['ref_image']
        rotate_raw_image = self._random_rotate(raw_image)
        rotate_ref_image = self._random_rotate(ref_image)
        return {'raw_image': rotate_raw_image,
                'ref_image': rotate_ref_image}


