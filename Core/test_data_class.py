#import data_class
import numpy as np
import torch 
import pandas as pd
csv = 'https://raw.githubusercontent.com/Xiar-fatah/Stock-Market-Patterns/master/Core/MSFT.csv'

#fetch = data_class.data(train_start = 0, train_end = 4000, test_start = 3980, test_end = 'last', window = 20, csv_path = csv)


def roll(start, end, window, df):
    data = []
    labels = []
    start = start + window # Begins at the first 20 elements
    if end == 'last':
        end = df.columns.shape[0]
        
    for i in range(start, end):
        temp = []
        for col in range(0, len(df.columns)): # (5004,16)
            if (df.columns[col] == 'date') == False | (df.columns[col] == '5. volume') == False:
                col_arr = df.iloc[:,col].tolist()    
                temp.append(np.reshape(col_arr[i-window:i], window))
        data.append(np.transpose(temp))
                
    for i in range(start, end): 
        close_arr = df['4. close'].tolist()
        labels.append(close_arr[i+window])

    return np.array(data), np.array(labels)


window = 20

df = pd.read_csv(csv).iloc[::-1]# Flipping the data
df = df.drop('date', 1)
df = df.drop('5. volume', 1)
df_mean = df.mean()[3]
df_std = df.std()[3]
df = (df-df.mean())/df.std()

# Train
x_train,y_train = roll(0, 4000, window,df)
check = y_train
x_train, y_train = torch.Tensor(x_train), torch.Tensor(y_train)

tensor_train = torch.utils.data.TensorDataset(x_train, y_train)

trainloader = torch.utils.data.DataLoader(tensor_train, batch_size=1,
                                         shuffle = True, num_workers=0)

# Test
x_test, y_test = roll(4000-window,'last', window, df)
x_test, y_test = torch.Tensor(x_test), torch.Tensor(y_test)
tensor_test = torch.utils.data.TensorDataset(x_test,y_test)

testloader = torch.utils.data.DataLoader(tensor_test, batch_size=1,
                                         shuffle = False, num_workers=0)

