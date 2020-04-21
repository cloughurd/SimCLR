import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd

class CompCars(Dataset):
    def __init__(self, data_root, encodings_loc, train=True, transform=None, label_type='make'):
        self.transform = transform
        self.data_root = data_root
        self.label_type = label_type
        if train:
            split_file = os.path.join(data_root, 'train.txt')
        else:
            split_file = os.path.join(data_root, 'test.txt')
        self.images = []
        with open(split_file, 'r') as f:
            for line in f:
                self.images.append(line.strip())
        attr_file = os.path.join(data_root, 'attributes.txt')
        self.attributes = pd.read_csv(attr_file, sep=' ', index_col=0)
        self.encodings = torch.load(encodings_loc)

    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        i = self.images[idx]
        encoded = self.encodings[idx]
        
        if self.label_type == 'make':
            label = int(i.split('/')[0]) - 1
        elif self.label_type == 'model':
            label = int(i.split('/')[1]) - 1
        elif self.label_type in self.attributes:
            model = int(i.split('/')[1])
            row = self.attributes.loc[model]
            label = row[self.label_type]
        else:
            label = -1

        return encoded, torch.LongTensor([label])
