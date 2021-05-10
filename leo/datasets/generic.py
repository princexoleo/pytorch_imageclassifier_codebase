"""
Generic datasets for generic datasets

"""

import torch

class GenericDataset:
    """Takes data, target & dtypes as a input """
    def __init__(self, data, targets, dtypes):
        self.data = data 
        self.targets = targets 
        self.dtypes = dtypes

    
    def __len__(self):
        return len(data)

    def __getitem__(self, index):
        data = self.data[index]
        targets = self.targets[index]

        return{
            "x": torch.tensor(data, dtype=self.dtypes[0]),
            "y": torch.tensor(targets, dtype=self.dtypes[1]),
        }