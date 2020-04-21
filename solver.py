import os
import json
import torch
import torch.nn as nn
from tqdm import tqdm

class CESolver:
    def __init__(self, model, train_loader, valid_loader, save_root, name='CESolver', device='cuda:0'):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.save_root = save_root
        self.name = name
        self.device = device
        self.train_loss = []
        self.train_acc = []
        self.valid_loss = []
        self.valid_acc = []

    def train(self, num_epochs=100, val_freq=1, save_freq=1):
        model = self.model.to(self.device)
        objective = nn.CrossEntropyLoss()
        optim = torch.optim.Adam(model.parameters())
        sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[
                                                        int(num_epochs * 0.3),
                                                        int(num_epochs * 0.5),
                                                        int(num_epochs * 0.8)])
        
        min_val_loss = float('inf')
        max_val_acc = 0
        for e in range(num_epochs):
            loop = tqdm(total=len(self.train_loader), position=0, leave=False)
            for x, y in self.train_loader:
                x = x.to(self.device)
                y = y.to(self.device).squeeze(1)

                optim.zero_grad()
                y_hat = model(x)
                loss = objective(y_hat, y)
                loss.backward()
                optim.step()

                acc = accuracy(y_hat, y)
                self.train_loss.append(loss.item())
                self.train_acc.append(accuracy(y_hat, y))
                loop.update(1)
                loop.set_description(f'Epoch: {e}, Loss: {loss.item()}, Accuracy: {acc}')

            if e % val_freq == 0:
                vl, va = self.validate(model, objective)
                if va > max_val_acc:
                    max_val_acc = va
                if vl < min_val_loss:
                    min_val_loss = vl
                    self.save(model, vl, e)
        
            sched.step()
        if (num_epochs-1) % val_freq != 0:
            vl, va = self.validate(model, objective)
            if va > max_val_acc:
                max_val_acc = va
            if vl < min_val_loss:
                min_val_loss = vl

        self.save(model, min_val_loss, 'final')
        print(f'Training completed with max validation accuracy of {max_val_acc}')
    
    def validate(self, model, objective):
        model.eval()
        vl_list = []
        va_list = []
        num_items = 0
        with torch.no_grad():
            for x, y in self.valid_loader:
                x = x.to(self.device)
                y = y.to(self.device).squeeze(1)
                y_hat = model(x)

                vl = objective(y_hat, y).item()
                batch_size = y.shape[0]
                vl_list.append(vl * batch_size)
                va = accuracy(y_hat, y)
                va_list.append(va * batch_size)
                num_items += batch_size

        total_vl = sum(vl_list) / num_items
        total_acc = sum(va_list) / num_items
        self.valid_loss.append((len(self.train_loss), total_vl))
        self.valid_acc.append((len(self.train_acc), total_acc))
        model.train()
        return total_vl, total_acc

    def save(self, model, vl, e):
        print(f'Saving {self.name} at epoch {e} with validation loss {vl}')
        if e == 'final':
            name = self.name = '-final'
        else:
            name = self.name
        model_file = os.path.join(self.save_root, f'{name}.pth')
        torch.save(model.state_dict(), model_file)
        loss_file = os.path.join(self.save_root, f'{self.name}.json')
        json.dump({
            'train_loss': self.train_loss,
            'train_acc': self.train_acc,
            'val_loss': self.valid_loss,
            'val_acc': self.valid_acc
        }, open(loss_file, 'w'))

def accuracy(output, target):
    output = output.argmax(dim=1)
    acc = output.eq(target).sum()
    acc = acc / float(target.numel())
    return acc.item()
