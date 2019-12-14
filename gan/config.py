from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn

class Config:
    def __init__(self, epochs, batch_size, learning_rateD, learning_rateG, device_num=0):
        self.name = "Diabetic Retinopathy with Uncertainty"
        self.device = torch.device("cuda:%s" % device_num if torch.cuda.is_available() else "cpu")
        self.device_name = torch.cuda.get_device_name(device_num) if torch.cuda.is_available() else "cpu"

        self.num_classes = 2
        self.num_workers = 10

        self.epochs = epochs
        self.batch_size = batch_size
        self.lrD = learning_rateD
        self.lrG = learning_rateG

        self.scheduler = None

        # GAN parameters
        # Number of channels in the training images. 
        self.nc = 3

        # Size of z latent vector (i.e. size of generator input)
        self.nz = 100
        
        # Size of feature maps in generator
        self.ngf = 64
        
        # Size of feature maps in discriminator
        self.ndf = 64
        
    def to_json(self):
        # write config to json so we can see what we did later
        pass

        

        
        
