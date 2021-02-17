import numpy as np
import torch
from skimage import io, transform

class Rescale(object):
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
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))
        return img

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

        return {'raw_image': torch.from_numpy(new_raw_image),
                'ref_image': torch.from_numpy(new_ref_image)}


