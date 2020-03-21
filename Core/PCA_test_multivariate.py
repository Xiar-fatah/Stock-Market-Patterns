import numpy as np
import pandas as pd
import data_class 
import matplotlib.pyplot as plt
# %%

#Import data
csv = 'https://raw.githubusercontent.com/Xiar-fatah/Stock-Market-Patterns/ADD_PCA/Core/Financial_Data/FORD_V2.csv'
data = data_class.data(train_start = 0, train_end = 4000, test_start = 3980, test_end = 'last', window = 20, csv_path = csv)
train_data = data.data_tot # Note that the data is already reversed from the class
# %%
#Standarize data and remove dates
def nor_date(df):
    df = df.drop('date', 1) # Remove date
    df = df.drop('5. volume', 1) # Remove volume
    df = (df-df.mean())/df.std() # Standarize
    return df, df.mean()[3], df.std()[3]

train_data_stand, mean, std = nor_date(train_data)
# %%
from sklearn.decomposition import PCA

# Choosing the number of components
pca_variance = PCA().fit(train_data_stand)
plt.plot(np.cumsum(pca_variance.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
# %%
# Perform PCA 
from sklearn.decomposition import PCA
pca = PCA(n_components=10) 

train_PCA = pca.fit_transform(train_data_stand)

# %%
# Create a multivariate rolling window
import torch 

def roll(start, end, window, df, close):
    if end == 'last':
        end = df.shape[0]
    data = [] # X
    labels = close[start + window:end] # Y
    start = start + window # Begins at the first 20 elements
    for i in range(start, end):
        temp = []
        fetch = []
        for col in range(0, df.shape[1]): 
            fetch = df[:,col].tolist() #fetch the column
            temp.append(np.reshape(fetch[i-window:i], window)) #Apply rolling window
        data.append(np.transpose(temp))
    return np.array(data), np.array(labels)

close = train_data_stand['4. close']

x_train, y_train = roll(0, 4000, 20, train_PCA, close)
x_test, y_test = roll(3980, 'last', 20, train_PCA, close)

# %% 
# Convert everything into tensors & load them into loaders
import torch 
x_train_tensor, y_train_tensor = torch.Tensor(x_train), torch.Tensor(y_train)
train = torch.utils.data.TensorDataset(x_train_tensor,y_train_tensor)
trainloader = torch.utils.data.DataLoader(train , batch_size=1,
                                              shuffle=True, num_workers=0)


x_test_tensor, y_test_tensor = torch.Tensor(x_test), torch.Tensor(y_test)
test = torch.utils.data.TensorDataset(x_test_tensor,y_test_tensor)
testloader = torch.utils.data.DataLoader(test , batch_size=1,
                                              shuffle=False, num_workers=0)

real = train_data_stand['4. close'].tolist()[4000:]








