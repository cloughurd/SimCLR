import argparse
import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from data_aug.stanfordcars import StanfordCarsMini, CarsDataset
from classification_loaders.encodedsc import EncodedStanfordCarsDataset
from models.resnet_simclr import ResNetSimCLR
from solver import CESolver

class FineTuner(nn.Module):
    def __init__(self, embedder, clf):
        super(FineTuner, self).__init__()
        self.embedder = embedder
        self.clf = clf

    def forward(self, x):
        h, _ = self.embedder(x)
        y_hat = self.clf(h)
        return y_hat

def run(config):
    model = ResNetSimCLR('resnet50', config.out_dim)
    model.load_state_dict(torch.load(config.model_file, map_location=config.device))
    model = model.to(config.device)
    clf = nn.Linear(2048, 196)

    train_data_dir = os.path.join(config.data_root, 'cars_train/')
    train_annos = os.path.join(config.data_root, 'devkit/cars_train_annos.mat')
    valid_data_dir = os.path.join(config.data_root, 'cars_test/')
    valid_annos = os.path.join(config.data_root, 'devkit/cars_test_annos_withlabels.mat')

    if config.encodings_file_prefix:
        train_dataset = EncodedStanfordCarsDataset(train_annos, config.encodings_file_prefix + '-train_encodings.pt')
        train_loader = DataLoader(train_dataset, batch_size=config.encodings_batch_size, shuffle=True)

        valid_dataset = EncodedStanfordCarsDataset(valid_annos, config.encodings_file_prefix + '-valid_encodings.pt')
        valid_loader = DataLoader(valid_dataset, batch_size=config.encodings_batch_size)

        tmp_clf = nn.Linear(2048, 196)
        clf_solver = CESolver(tmp_clf, train_loader, valid_loader, config.save_root, name=config.name+'-clf', device=config.device)
        clf_solver.train(config.encodings_num_epochs)
        clf_filename = os.path.join(config.save_root, f'{config.name}-clf.pth')
        clf.load_state_dict(torch.load(clf_filename, map_location=config.device))

    full = FineTuner(model, clf)

    t = T.Compose([
            T.Resize(512),
            T.CenterCrop(512),
            T.ToTensor(),
            T.Lambda(lambda x: x.repeat(3,1,1) if x.shape[0] == 1 else x)
        ])
    train_dataset = StanfordCarsMini(train_annos, train_data_dir, t)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    valid_dataset = CarsDataset(valid_annos, valid_data_dir, t)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size)

    solver = CESolver(full, train_loader, valid_loader, config.save_root, name=config.name, device=config.device)
    full = full.to(config.device)
    print(solver.validate(full, nn.CrossEntropyLoss()))
    # solver.train(config.num_epochs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, help='location of model to fine tune')
    parser.add_argument('--data_root', type=str, default='/multiview/datasets/StanfordCars')
    parser.add_argument('--save_root', type=str, default='./finetuned', help='location to save the fine tuned model')
    parser.add_argument('--num_epochs', type=int, default=80, help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--out_dim', type=int, default=256, help='dimension of projection')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--name', type=str, default='SCMTuned')
    parser.add_argument('--encodings_file_prefix', type=str, default='./encodings/stanfordCars-scmodel')
    parser.add_argument('--encodings_batch_size', type=int, default=100)
    parser.add_argument('--encodings_num_epochs', type=int, default=160)

    args = parser.parse_args()
    run(args)