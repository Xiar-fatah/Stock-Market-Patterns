import numpy as np
import torch 
import pandas as pd

class data:
    
    def __init__(self, train_start, train_end, validation_start, validation_end, test_start, test_end, window, csv_path):
        self.train_start, self.train_end = train_start, train_end
        self.test_start, self.test_end = test_start, test_end
        self.validation_start, self.validation_end = validation_start, validation_end
        self.window = window
        self.csv_path = csv_path
        self.data_tot = self.fetch(self.csv_path)
        self.real = self.stock(self.data_tot, self.test_start, self.test_end,self.window)
        self.train_x, self.train_y = self.roll(self.train_start, self.train_end, self.window, self.data_tot)
        self.validation_x, self.validation_y = self.roll(self.validation_start, self.validation_end, self.window, self.data_tot)
        self.test_x, self.test_y = self.roll(self.test_start, self.test_end, self.window, self.data_tot)
        self.shuffle_train = True
        self.shuffle_test = False
        self.trainloader = self.arr_tensor(self.train_x, self.train_y, self.shuffle_train)
        self.validationloader = self.arr_tensor(self.validation_x, self.validation_y, self.shuffle_train)
        self.testloader = self.arr_tensor(self.test_x, self.test_y, self.shuffle_test)
        self.df_mean, self.df_std = self.normalize(self.data_tot) 
        
    def normalize(self,df):
        # store the standardize constants
        df = df.drop('date', 1) # Remove date
        return df.mean()[3], df.std()[3]
    
    def fetch(self, csv_path):    
        csv_data = pd.read_csv(csv_path)# Read in the data and flip it
        Bool,i = True,0
        while Bool:
            if csv_data.iloc[i,0] == '2020-02-21':
                csv_data = csv_data[i:]
                Bool = False
            i += 1
            
        Bool,i = True,len(csv_data)
        while Bool:
            if csv_data.iloc[i-1,0] == '2000-05-05':
                csv_data = csv_data[:i]
                Bool = False
            i -= 1
        return csv_data.iloc[::-1]# Read in the data and flip it

    
    def stock(self, df, start, end, window):
        if end == 'last':
            end = df.columns.shape[0]
        return df['4. close'].tolist()[start + window:end]
        
    def roll(self, start, end, window, df):
        df = df.drop('date', 1) # Remove date
        if end == 'last':
            end = len(df) # length of rows
        df = (df-df.mean())/df.std()
        data = [] # X
        close_arr = df['4. close'].tolist()
        labels = close_arr[start + window:end] # Y
        start = start + window # Begins at the first 20 elements
        for i in range(start, end):
            temp = []
            for col in range(0, len(df.columns)): 
                col_arr = df.iloc[:,col].tolist()
                temp.append(np.reshape(col_arr[i-window:i], window))
            data.append(np.transpose(temp))
        return np.array(data), np.array(labels)
        
    def arr_tensor(self, x, y, shuffle):
        x, y = torch.Tensor(x), torch.Tensor(y)
        tensor = torch.utils.data.TensorDataset(x,y)
        return torch.utils.data.DataLoader(tensor, batch_size=1,
                                                 shuffle = shuffle, num_workers=0)

        
        
        
        
csv = 'https://raw.githubusercontent.com/Xiar-fatah/Stock-Market-Patterns/ADD_PCA/Core/Financial_Data/FORD.csv'
data = data(train_start = 0, train_end = 4000, validation_start = 3980, validation_end = 4500,
            test_start = 4480, test_end = 'last', window = 20, csv_path = csv)        
        
        
        
        
        
        
        
        