import os
import pickle
import numpy as np
from pathlib import Path
from collections import OrderedDict
from torchvision import datasets, transforms

class MINISTDataset():

    def __init__(self, opt, phase="train", **kwargs):

        # path
        self.opt = opt
        self.phase = phase
        self.path_root = Path(opt.DATASET.root) 
                                  
    
    def get_train_dataset(self):
        return datasets.MNIST(root=str(self.path_root),
                                       train=True,
                                       transform=transforms.ToTensor(),
                                       download=True)

    def get_valid_dataset(self):
        return datasets.MNIST(root=str(self.path_root),
                              train=False,
                              transform=transforms.ToTensor())

    def get_test_dataset(self):
        raise("NOT Implement!")