import numpy as np
import pandas as pd
import data_class 
import matplotlib.pyplot as plt
# %%

#Import data
csv = 'https://raw.githubusercontent.com/Xiar-fatah/Stock-Market-Patterns/ADD_PCA/Core/Financial_Data/FORD.csv'
data = data_class.data(train_start = 0, train_end = 4000, test_start = 3980, test_end = 5003, window = 20, csv_path = csv)
train_data = data.data_tot # Note that the data is already reversed from the class
# %%
#Standarize data and remove dates
def nor_date(df):
    df = df.drop('date', 1) # Remove date
    df = df.drop('5. volume', 1) # Remove volume
    df = (df-df.mean())/df.std() # Standarize
    return df

train_data = nor_date(train_data)
# %%
# For now we only want low, high, closing and opening to apply PCA to
PCA_df = pd.DataFrame()
PCA_df['open'] = train_data['1. open']
PCA_df['high'] = train_data['2. high']
PCA_df['low'] = train_data['3. low']
PCA_df['close'] = train_data['4. close']

# %%
# Perform PCA 
from sklearn.decomposition import PCA
pca = PCA(n_components=1) # We go from 4 dimensional to 1 dimensional
train_PCA = pca.fit_transform(PCA_df)
# %%
# Plot values
plt.plot(PCA_df['open'].tolist()), plt.plot(PCA_df['high'].tolist()),
plt.plot(PCA_df['low'].tolist()), plt.plot(PCA_df['close'].tolist()),
plt.plot(train_PCA)
plt.legend(('open','high','low','close','1DPCA'))
plt.ylabel(('Standarized Dollars'))
plt.xlabel(('Time'))

# %%
# Create rolling window
import torch 
def univariate_data(dataset,labels_PCA, start_index, end_index, window):
    # Store data and labels
    data = []
    labels = []
    
    start_index = start_index + window
    if end_index is None:
        end_index = len(dataset)
    
    for i in range(start_index, end_index):
        indices = range(i-window, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (window, 1)))
        labels.append(labels_PCA[i])
    return np.array(data), np.array(labels)

 # Store the labels of last 1000 days
window = 20
TRAIN_SPLIT = 4000
labels = PCA_df['close'].tolist()
x_train, y_train = univariate_data(train_PCA, labels, 0, TRAIN_SPLIT, window)

x_test, y_test = univariate_data(train_PCA, labels, TRAIN_SPLIT, None, window)

x_train, y_train = torch.Tensor(x_train), torch.Tensor(y_train)
train = torch.utils.data.TensorDataset(x_train,y_train)

trainloader = torch.utils.data.DataLoader(train , batch_size=1,
                                              shuffle=True, num_workers=0)

x_test, y_test= torch.Tensor(x_test), torch.Tensor(y_test)
test = torch.utils.data.TensorDataset(x_test,y_test)

testloader = torch.utils.data.DataLoader(test , batch_size=1,
                                              shuffle=False, num_workers=0)


real = data.real # Fetching the 1000 last values
mean = data.mean
std = data.std
