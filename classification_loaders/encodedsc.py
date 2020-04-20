import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io

class EncodedStanfordCarsDataset(Dataset):
    def __init__(self, mat_loc, encodings_loc):
        self.full_data_set = scipy.io.loadmat(mat_loc)
        self.car_annotations = self.full_data_set['annotations'][0]
        self.encodings = torch.load(encodings_loc)


    def __len__(self):
        return len(self.car_annotations)

    def __getitem__(self, idx):
        encoded = self.encodings[idx]
        car_class = self.car_annotations[idx][-2][0][0] -1
        y = torch.LongTensor([car_class])
        return encoded, y