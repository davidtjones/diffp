import torch
# from matplotlib.pyplot import imread
from PIL import Image

class LoadImage(object):
    def __init__(self, data_root):
        self.data_root = data_root
    def __call__(self, image_name):
        # load image to be used in training
        # img = imread(self.data_root / image_name)
        img = Image.open(self.data_root / image_name)  # load as PIL
        return img

class Norm(object):
    def __call__(self, img):
        # normalize between -1 and 1:
        img = (img/255)
        img = img*2
        img = img-1
        return img




        
        
