# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 17:58:30 2024

@author: theox
"""
import numpy as np
from torch.utils.data import Dataset

class NeuralDataset(Dataset):
    def __init__(self, activity, labels, transform=None):
        self.activity  = activity
        self.labels    = labels
        self.transform = transform
    def __getitem__(self, idx):
#         sample = np.array(self.activity[idx,:]), np.array(self.labels[idx])
        sample = np.array(self.activity[idx,:]), np.array(self.labels[idx,:])
        if self.transform:
            sample  = self.transform(sample)
        return sample
    def __len__(self):
        return len(self.labels)