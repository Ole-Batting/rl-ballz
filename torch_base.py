import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


class Funnel(nn.Module):
    def __init__(self, input_size, n_layers, output_size):
        super(Funnel, self).__init__()

        self.dims = np.linspace(input_size, output_size, n_layers, dtype=int, endpoint=False)
        layers = []

        for hi, ho in zip(self.dims[:-1], self.dims[1:]):
            layers.append(nn.Linear(hi, ho))
            layers.append(nn.ELU())
            #layers.append(nn.Dropout1d())
        layers.append(nn.Linear(self.dims[-1], output_size))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.net(x)
        return x


class ImageFunnel(nn.Module):
    def __init__(self, ch_in, ch_out, width, height, n_layers):
        super(ImageFunnel, self).__init__()

        layers = []
        #out_width = width
        #out_height = height

        for i in range(n_layers):
            layers.append(nn.Conv2d(ch_in if i==0 else ch_out, ch_out, 3, padding=1))
            layers.append(nn.ELU())
            layers.append(nn.Dropout2d())
            #layers.append(nn.MaxPool2d(2))
            #out_width = np.floor((out_width - 2) / 2)
            #out_height = np.floor((out_height - 2) / 2)
        
        self.net = nn.Sequential(*layers)
        self.output_dim = int(ch_out * width * height)
    
    def forward(self, x):
        x = self.net(x)
        return x.view(-1, self.output_dim)
    

class DeepQNetwork(nn.Module):
    def __init__(self, ch_in, ch_out, width, height, n_layers_cnn, n_layers_fcn, output_dim):
        super(DeepQNetwork, self).__init__()
        self.cnn = ImageFunnel(ch_in, ch_out, width, height, n_layers_cnn)
        self.net = Funnel(self.cnn.output_dim, n_layers_fcn, output_dim)
        self.lin = nn.Linear(output_dim, output_dim)
    
    def forward(self, x):
        x = self.cnn(x)
        x = self.net(x)
        x = self.lin(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, ch_in, ch_out, width, height, n_layers_cnn, n_layers_fcn, output_dim):
        super(PolicyNetwork, self).__init__()
        self.cnn = ImageFunnel(ch_in, ch_out, width, height, n_layers_cnn)
        self.net = Funnel(self.cnn.output_dim, n_layers_fcn, output_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        x = self.cnn(x)
        x = self.net(x)
        x = self.softmax(x)
        return x


class ValueNetwork(nn.Module):
    def __init__(self, ch_in, ch_out, width, height, n_layers_cnn, n_layers_fcn):
        super(ValueNetwork, self).__init__()
        self.cnn = ImageFunnel(ch_in, ch_out, width, height, n_layers_cnn)
        self.net = Funnel(self.cnn.output_dim, n_layers_fcn, 1)
    
    def forward(self, x):
        x = self.cnn(x)
        x = self.net(x)
        return x
