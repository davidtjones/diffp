from pathlib import Path
import pandas as pd

class Config:
    def __init__(self, epochs, batch_size, learning_rate, optimizer, criterion, scheduler=None, device_num=0):
        self.name = "Diabetic Retinopathy with Uncertainty"
        self.device = torch.device("cuda:%s" % device_num if torch.cuda.is_available() else "cpu")
        self.device_name = torch.cuda.get_device_name(device_num) if torch.cuda.is_available() else "cpu"

        self.num_classes = 2
        self.num_workers = 8

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.optim = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        

    def to_json(self):
        # write config to json so we can see what we did later
        pass

        

        
        
