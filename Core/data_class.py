import numpy as np
import torch 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class data:
    
    def __init__(self, train, test, window, csv_path):
        window = self.window
        tot_data = self.fetch(csv_path)
        train_data, test_data = self.roll(train, window, tot_data), 
        self.roll(test, window, tot_data)
        
    def fetch(self, csv_path):   
        return pd.read_csv(csv_path)
    
    def roll(self, start, window, data):
        # Store data and labels
        data = []
        labels = []
        
        start_index = start_index + history_size
        if end_index is None:
            end_index = len(dataset) - target_size
        
        for i in range(start_index, end_index):
            indices = range(i-history_size, i)
            # Reshape data from (history_size,) to (history_size, 1)
            data.append(np.reshape(dataset[indices], (history_size, 1)))
            labels.append(dataset[i+target_size])
        return np.array(data), np.array(labels)
        
        