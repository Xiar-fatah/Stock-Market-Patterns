"""
    This file is used for testing purpose. It is preparing the closing price for microsoft stock.
"""

import numpy as np
import pandas as pd
import torch 

#csv_path = 'https://raw.githubusercontent.com/Xiar-fatah/Stock-Market-Patterns/master/Core/MSFT.csv'
csv_path = 'https://raw.githubusercontent.com/Xiar-fatah/Stock-Market-Patterns/ADD_PCA/Core/Financial_Data/FORD.csv'
df = pd.read_csv(csv_path)

def univariate_data(dataset, start_index, end_index, history_size, target_size):
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

uni_data = df['4. close']
uni_data = np.flip(uni_data.tolist())
real = uni_data[4020:]
mean, std = uni_data.mean(),uni_data.std()
uni_data = (uni_data-mean)/std

window = 20
prediction = 0
TRAIN_SPLIT = 4000
x_train, y_train = univariate_data(uni_data, 0, TRAIN_SPLIT,
                                           window,
                                           prediction)

x_test, y_test = univariate_data(uni_data, TRAIN_SPLIT, None,
                                       window,
                                       prediction)

x_train, y_train = torch.Tensor(x_train), torch.Tensor(y_train)
train = torch.utils.data.TensorDataset(x_train,y_train)

trainloader = torch.utils.data.DataLoader(train , batch_size=1,
                                              shuffle=True, num_workers=0)

x_test, y_test= torch.Tensor(x_test), torch.Tensor(y_test)
test = torch.utils.data.TensorDataset(x_test,y_test)

testloader = torch.utils.data.DataLoader(test , batch_size=1,
                                              shuffle=False, num_workers=0)



















































