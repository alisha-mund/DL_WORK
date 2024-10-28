from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
import pandas as pd

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]

class ChallengeDataset(Dataset):
    def __init__(self, data, mode):
        super().__init__()

        self.data = data
        self.mode = mode
        self.transform = {
            'train': tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(train_mean,train_std)]),

            'val': tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(train_mean,train_std)])

        }


    def __len__(self):
        datalen = len(self.data)
        return datalen

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        img = self.data.iloc[index,0]
        img = self.transform[self.mode](gray2rgb(imread(img)))
        lb1 = self.data.iloc[index,1]
        lb2 = self.data.iloc[index,2]
        fl_data = (img, torch.tensor([lb1,lb2]))
        return fl_data