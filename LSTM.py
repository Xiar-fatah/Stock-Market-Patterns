import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, T, logger):
        super(LSTM,self).__init()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = T
        self.logger = logger
        
        self.lstm_layer = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = 1)
        self.attn_linear = nn.Linear(in_features = 2 * hidden_size + T - 1, out_features = 1)
    
    def forward(self, t)
        


if __name__ == "__main__":
