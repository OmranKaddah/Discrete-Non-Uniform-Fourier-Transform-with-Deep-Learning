# *_*coding:utf-8 *_*
import os
import numpy as np
import torch
from data_utils.data_loader import loadData
from torch.utils.data import Dataset

class nufftDataLoader(Dataset):
    def __init__(self, numSamples, cat="train",isFlattened =False,isFlattenedIn=False):
        self.x  = None
        self.y  = None
        
        if cat is "train":
            (self.x,self.y),_ = loadData(numSamples,0,isFlattened,isFlattenedIn)
        if cat is "test":
            _, (self.x,self.y) = loadData(0,numSamples,isFlattened,isFlattenedIn)
        self.len = self.x.shape[0]
        self.x = self.x / np.max(self.x)
        self.x = torch.from_numpy(self.x)
        self.y = torch.from_numpy(self.y)
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x[index], self.y[index]

