import torch
from PIL import Image

class LoadImage(object):
    def __init__(self, data_root):
        self.data_root = data_root
    def __call__(self, image_name):
        # load image to be used in training
        img = Image.open(self.data_root / image_name)  # load as PIL
        return img


