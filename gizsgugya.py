#### conda install -n pytorch_cpu ipykernel --update-deps --force-reinstall
#### conda install -n pytorch_cpu matplotlib tqdm numpy Pillow 
#### conda install -n pytorch_cpu pytorch torchvision torchaudio cpuonly -c pytorch

#%%

%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import io
import numpy as np
import random
import requests
from PIL import Image
import PIL.ImageOps    
from tqdm import tqdm

import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

plt.rcParams.update({'figure.max_open_warning': 0})


#%%

def load_numpy_arr_from_url(url):
    """
    Loads a numpy array from surfdrive. 
    
    Input:
    url: Download link of dataset 
    
    Outputs:
    dataset: numpy array with input features or labels
    """
    
    response = requests.get(url)
    response.raise_for_status()

    return np.load(io.BytesIO(response.content)) 
    
    
    
#Downloading may take a while..
train_data = load_numpy_arr_from_url('https://surfdrive.surf.nl/files/index.php/s/4OXkVie05NPjRKK/download')
train_label = load_numpy_arr_from_url('https://surfdrive.surf.nl/files/index.php/s/oMLFw60zpFX82ua/download')

print(f"train_data shape: {train_data.shape}")
print(f"train_label shape: {train_label.shape}\n")


#%%

def plot_case(caseID,train_data,labels):
    """
    Plots a single sample of the query dataset
    
    Inputs
    caseID: Integer between 0 and 99, each corresponding to a single sample in the query dataset 
    """
    

    support_set,queries = np.split(train_data, [5], axis=1)
    
    f, axes = plt.subplots(1, 6, figsize=(20,5))

    # plot anchor image
    axes[5].imshow(queries[caseID, 0])
    axes[5].set_title(f"Query image case {caseID}", fontsize=15)

    # show all test images images 
    [ax.imshow(support_set[caseID, i]) for i, ax in enumerate(axes[0:-1])]


    # Add the patch to the Axes
    for ind in np.where(labels[caseID]==True)[0]:
        axes[ind].add_patch(Rectangle((0,0),27,27,linewidth=2, edgecolor='r',facecolor='none'))


    [plot_case(caseID,train_data,train_label) for caseID in range(5)]

#%%

test_data = load_numpy_arr_from_url('https://surfdrive.surf.nl/files/index.php/s/06c34QVUr69CxWY/download')
test_label = load_numpy_arr_from_url('https://surfdrive.surf.nl/files/index.php/s/LQIH1CW7lfDXevk/download')

print(f"test_data shape: {test_data.shape}")
print(f"test_label shape: {test_label.shape}\n")

[plot_case(caseID,test_data,test_label) for caseID in range(5)]

#%%

class MyDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.FloatTensor(data)
        self.targets = torch.FloatTensor(targets)

        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        

        return x, y
    
    def __len__(self):
        return len(self.data)

#%%

train_dataset=MyDataset(train_data,train_label)
test_dataset=MyDataset(test_data,test_label)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10)

print(train_data.shape)

#%%

## change it to triplet/consdjaslkdjasl loss
def loss_batch(model, loss_func, xb, yb, opt=None):
    
    output=model(xb)
    loss = loss_func(output, yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    
    return loss.item(), len(xb)


#%%

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    logs = {}
    # liveloss = PlotLosses()
    
    for epoch in tqdm(range(epochs)):
        # training process
        model.train()

        running_loss = 0.0
        running_corrects = 0
        sample_num=0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            
            # forward
            # backward and optimize only if in training phase
            losses, nums = loss_batch(model, loss_func, xb, yb,opt)
            
            # statistics
            running_loss += losses * xb.size(0)
            sample_num+=nums
            
        train_loss = running_loss / sample_num
        logs['loss'] = train_loss
        
        model.eval()
        with torch.no_grad():
            running_loss = 0.0
            sample_num=0
            for xb, yb in valid_dl:
                xb, yb = xb.to(device), yb.to(device)
                
                # forward
                losses, nums = loss_batch(model, loss_func, xb, yb)
                
                # statistics
                running_loss += losses * xb.size(0)
                sample_num+=nums

            val_loss = running_loss / sample_num
            logs['val_loss'] = val_loss
        
        # liveloss.update(logs)
        # liveloss.draw()
            
        # print the results
        print(
            f'EPOCH: {epoch+1:0>{len(str(epochs))}}/{epochs}',
            end=' '
        )
        print(f'LOSS: {train_loss:.4f}', end=' ')
        print(f'VAL-LOSS: {val_loss:.4f}',end='\n')

#%%

## change to siamese network implementation
class Magic(nn.Module):
  def __init__(self):
    super(Magic, self).__init__()
    self.layers = nn.Sequential(
        nn.Conv2d(6, 32, 3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(0.2),
        nn.Conv2d(32, 64, 3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(0.2),
        nn.Flatten(),
        nn.Linear(1600, 5),
        nn.Sigmoid()
    )

  def forward(self, x):
    x = self.layers(x)
    return x


#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%
def get_mlpmodel():
  model = Magic()
  return model, optim.Adam(model.parameters())

#%%
model, opt = get_mlpmodel()

model.to(device)

fit(100, model, nn.BCELoss(), opt, train_loader, test_loader)
# %%
