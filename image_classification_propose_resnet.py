import os, glob, argparse, tqdm
from PIL import Image

import numpy as np 
import pandas as pd

import torch
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

import matplotlib.pyplot as plt 

#BASE_DIR = os.path.dirname(__file__)
BASE_DIR = os.getcwd()

DATA_PATH = os.path.join(BASE_DIR, 'preprocessed_data')


JAY_TRAIN_PATH = os.path.join(DATA_PATH, 'train', 'jaywalks')
JAY_TEST_PATH = os.path.join(DATA_PATH, 'test', 'jaywalks')

OHR_TRAIN_PATH = os.path.join(os.path.split(DATA_PATH)[0], 'train', 'others')
OHR_TEST_PATH = os.path.join(os.path.split(DATA_PATH)[0], 'test', 'others')


parser = argparse.ArgumentParser(description='ml_final_assignment')

parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--input_size', type=tuple, default=(224, 224))
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
parser.add_argument('--num_epochs', type=int, default=30)
args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

class ImageTransform():
    def __init__(self, input_size = args.input_size):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.Resize(input_size),
                transforms.ToTensor()
            ]),
            'test': transforms.Compose([
                transforms.Resize(input_size), 
                transforms.ToTensor()
            ])
        }

    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)

def make_datapath_list(phase='train'):
    target_path = os.path.join(DATA_PATH, phase, '**','*.jpg')

    path_list = []

    for path in glob.glob(target_path):
        path_list.append(path)

    return path_list 

class img_datasets(Dataset):
    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list 
        self.transform = transform 
        self.phase = phase 


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)

        img_transformed = self.transform(img, self.phase)

        if 'jaywalk' in img_path:
            label = 1
        
        elif 'other' in img_path:
            label = 0 
        
        return img_transformed, label

train_list = make_datapath_list(phase='train')
val_list = make_datapath_list(phase='test')


train_dataset = img_datasets(file_list=train_list, transform=ImageTransform(), phase='train')
val_dataset = img_datasets(file_list=val_list, transform=ImageTransform(), phase='test')


train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

model = models.resnet50(pretrained=False).to(device)
model.fc = nn.Linear(in_features = 2048, out_features=2).to(device)
model
params_to_update = []
update_param_names = ['fc.weight', 'fc.bias']

for name, param in model.named_parameters():
    if name in update_param_names:
        param.requires_grad = True 
        
        params_to_update.append(param)
    
    else:
        param.requires_grad = False


#optimizer = optim.Adam(params=params_to_update, lr = args.learning_rate)
optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()


def train(model, dataloader, criterion, optimizer):
    model.train()

    train_batch_losses = 0
    train_batch_acces = 0

    for batch_id, (imgs, labels) in enumerate(dataloader):
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs).to(device)
        loss = criterion(outputs, labels).to(device)

        preds = torch.argmax(outputs, dim=1)

        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        batch_acc = ((preds == labels.data).sum() / len(labels)).item()
        print(f' train batch [{batch_id + 1}/{len(dataloader)}] train batch loss {batch_loss:.4f}, train batch acc: {batch_acc*100:.2f}%')

        train_batch_losses += batch_loss
        train_batch_acces += batch_acc
    
    train_batch_losses /= len(dataloader)
    train_batch_acces /= len(dataloader)

    return train_batch_losses, train_batch_acces


def evaluate(model, dataloader, criterion):
    test_batch_losses = 0
    test_batch_acces = 0

    model.eval()
    for batch_id, (imgs, labels) in enumerate(dataloader):
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs).to(device)
        loss = criterion(outputs, labels)

        preds = torch.argmax(outputs, dim=1)

        batch_loss = loss.item()
        batch_acc = ((preds == labels).sum() / len(labels)).item()
        print(f' test batch [{batch_id +1}/{len(dataloader)}] test batch loss {batch_loss:.4f}, test batch acc: {batch_acc*100:.2f}%')

        test_batch_losses += batch_loss 
        test_batch_acces += batch_acc 
    
    test_batch_losses /= len(dataloader)
    test_batch_acces /= len(dataloader)

    return test_batch_losses, test_batch_acces




if __name__ == '__main__':
    train_epoch_losses, train_epoch_acces = [], []
    test_epoch_losses, test_epoch_acces = [], []
    min_loss = float('inf')

    for epoch in tqdm.tqdm(range(args.num_epochs)):
        train_batch_losses, train_batch_acces = train(model, train_dataloader, criterion, optimizer)
        test_batch_losses, test_batch_acces = evaluate(model, val_dataloader, criterion)
#        if min_loss > test_batch_losses:
#            torch.save(model.state_dict(), 'propose_resnet_model.pt') ##########################
#            min_loss = test_batch_losses

        train_epoch_losses.append(train_batch_losses)
        train_epoch_acces.append(train_batch_acces)

        test_epoch_losses.append(test_batch_losses)
        test_epoch_acces.append(test_batch_acces)


    results = pd.DataFrame([train_epoch_losses, train_epoch_acces, test_epoch_losses, test_epoch_acces]).T
    results.columns =['train_loss', 'train_acc', 'test_loss', 'test_acc']
    results.to_csv('result_propose_resnet_non_pretrained.csv') ##################################



fig, axes = plt.subplots(nrows = 2, figsize = (16, 9))
axes[0].plot(np.arange(1, args.num_epochs + 1), train_epoch_losses, label='train_loss')
axes[0].plot(np.arange(1, args.num_epochs + 1), test_epoch_losses, label='test_loss')
axes[0].set_xticks(np.arange(1, args.num_epochs + 1))
axes[0].legend()


axes[1].plot(np.arange(1, args.num_epochs + 1), train_epoch_acces, label='train_accuracy')
axes[1].plot(np.arange(1, args.num_epochs + 1), test_epoch_acces, label='test_accuracy')
axes[1].set_xticks(np.arange(1, args.num_epochs + 1))
axes[1].legend()


axes[0].set_xlabel('EPOCHS')
axes[1].set_xlabel('EPOCHS')
plt.show()


results = pd.DataFrame([train_epoch_losses, train_epoch_acces, test_epoch_losses, test_epoch_acces], index = ['train_loss', 'train_acc', 'test_loss', 'test_acc']).T

