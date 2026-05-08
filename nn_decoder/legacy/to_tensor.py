# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 17:56:12 2024

@author: theox
"""

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

class ToTensor:
    # Convert ndarrays to Tensors
    def __call__(self, sample):
        inputs, targets = sample
#         return torch.from_numpy(inputs).to(torch.float32).to(device), torch.from_numpy(targets).type(torch.long).to(device)
        return torch.from_numpy(inputs).to(torch.float32).to(device), torch.from_numpy(targets).to(torch.float32).to(device)