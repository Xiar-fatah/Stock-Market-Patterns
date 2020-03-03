import numpy as np
import torch 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class data:
    
    def __init__(self, train, test, window, csv_path):
        self.window = window
        self.csv_path = csv_path
        self.data = self.fetch(self.csv_path)
#        self.train_data, self.test_data = self.roll(self.train, self.test, self.window, self.tot_data), 
        
    def fetch(self, csv_path):   
        return pd.read_csv(csv_path)
    
#    def roll(self, start, end, window, data):
#        # Store data and labels
#        data = []
#        labels = []
#        col_len = data.shape[0] #(5004,16)
#        
#        start = start + window
#        for i in range(start, end_index):
#            indices = range(i-history_size, i)
#            # Reshape data from (history_size,) to (history_size, 1)
#            data.append(np.reshape(dataset[indices], (history_size, 1)))
#            labels.append(dataset[i+target_size])
#        return (np.array(train_data), np.array(train_labels)), 
#            (np.array(test_data), np.array(test_labels))
        
        