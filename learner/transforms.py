import torch
from matplotlib.pyplot import imread

class LoadImage(object):
    def __init__(images_path):
        self.images_path = images_path
        
    def __call__(self, image_path):
        # load image to be used in training
        img = imread(image_path)
        return img




        
        
