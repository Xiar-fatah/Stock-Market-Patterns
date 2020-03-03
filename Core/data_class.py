import numpy as np
import torch 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class data:
    
    def __init__(self, train_start, train_end, test_start, test_end, window, csv_path):
        self.train_start, self.train_end = train_start, train_end
        self.test_start, self.test_end = test_start, test_end
        self.window = window
        self.csv_path = csv_path
        self.train_x, self.train_y = self.roll(self.train_start, self.train_end, self.window, self.csv_path)
        self.test_x, self.test_y = self.roll(self.test_start, self.test_end, self.window, self.csv_path)
        self.shuffle_train = True
        self.shuffle_test = False
        self.dataloader_train = self.arr_tensor(self.train_x, self.train_y, self.shuffle_train)
        self.dataloader_test = self.arr_tensor(self.test_x, self.test_y, self.shuffle_test)
        
    def roll(self, start, end, window, path):
        data = []
        labels = []
        data_tot = pd.read_csv(path).iloc[::-1]
        start = start + window # Begins at the first 20 elements
        if end == 'last':
            end = data_tot.columns.shape[0]
            
        for i in range(start, end):
            temp = []
            for col in range(0, len(data_tot.columns)): # (5004,16)
                if (data_tot.columns[col] == 'date') == False | (data_tot.columns[col] == '5. volume') == False:
                    col_arr = data_tot.iloc[:,col].tolist()    
                    temp.append(np.reshape(col_arr[i-window:i], window))
            data.append(np.transpose(temp))
                    
        for i in range(start, end): 
            close_arr = data_tot['4. close'].tolist()
            labels.append(close_arr[i-window:i])
    
        return np.array(data), np.array(labels)
        
    def arr_tensor(self, x, y, shuffle):
        x, y = torch.Tensor(x), torch.Tensor(y)
        tensor = torch.utils.data.TensorDataset(x,y)
        return torch.utils.data.DataLoader(tensor, batch_size=1,
                                                 shuffle = shuffle, num_workers=0)

        