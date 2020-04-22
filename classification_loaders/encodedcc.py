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
        self.attr_label = False
        if label_type in self.attributes:
            self.attr_label = True
            self.attributes = self.attributes[self.attributes[label_type] != 0]
            self.labels = self.attributes[label_type].to_dict()
            self.trimmed = []
            for idx in range(len(self.images)):
                i = self.images[idx]
                model = int(i.split('/')[1])
                if model in self.labels:
                    self.trimmed.append(idx)
        self.encodings = torch.load(encodings_loc)

    def __len__(self):
        if self.attr_label:
            return len(self.trimmed)
        else:
            return len(self.images)
        
    def __getitem__(self, idx):
        if self.attr_label:
            idx = self.trimmed[idx]
            i = self.images[idx]
            x = self.encodings[idx]
            model = int(i.split('/')[1])
            label = self.labels[model] - 1
        else:
            i = self.images[idx]
            x = self.encodings[idx]
            if self.label_type == 'make':
                label = int(i.split('/')[0]) - 1
            elif self.label_type == 'model':
                label = int(i.split('/')[1]) - 1
            else:
                label = -1

        return x, torch.LongTensor([label])
