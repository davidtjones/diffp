import os
from torch.utils.tensorboard import SummaryWriter

class TBoard():
    def __init__(self, results_dir):
        print("Started SummaryWriter")
        self.writer = SummaryWriter(log_dir=results_dir)

    def __del__(self):
        self.writer.close()

    def add_to_metrics_plot(self, epoch, metrics):
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, epoch)

    def add_image(self, name, img, step):
        self.writer.add_image(name, img, step)
