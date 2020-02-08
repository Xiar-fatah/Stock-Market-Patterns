import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

#Importing data
url = 'https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv'
dataset_train = pd.read_csv(url)
#Creating training dataset by extracting opening values of daa_set_train
train_data = dataset_train.iloc[:, 1:2].values

#Machine learning models adapt better to values between 0 and 1
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
train_data_scaled = sc.fit_transform(train_data)

#The input for LSTM is a three dimensional tensor
#Creating a data structure with 60 timesteps and 1 output
X_train = [] #Samples
y_train = [] #Labels
for i in range(60, len(train_data_scaled)):
    X_train.append(train_data_scaled[i-60:i, 0])
    
    y_train.append(train_data_scaled[i, 0])
"""
    What we have done is extract the opening values from 2010 to 2018.
    Put them into train_data and scaled them into train_data_scaled.
    Now we put the first 59 elements of opening price into X_train,
    and the last element in y_train, thus representing samples and
    labels. Therefore X_train is now an array with the shape 1975
    rows and 60 columns, where each row represents a prediction.
    However, the input needs to a three dimensional tensor.
    Hence a transformation is needed. The tensor according
    to documentations need to be in the form of
    (sequence, batch, features). Note the 
    batch size is 60,
    the amount of days is 1975, thus time step is 1975,
    the only feature we have is opening price, input_dim = 1
    
"""
#reshape into an numpy array
#X_train, y_train = np.array(X_train), np.array(y_train) 
X_train, y_train = torch.Tensor(X_train), torch.Tensor(y_train)
#Reshaping into three dimensional tensor

"""
    numpy.reshape(a, newshape, order='C'),
    gives a new shape to an array without changing its data.
    X_train.shape[0], X_train.shape[1] = 60,1975
"""
test = []
for i in range(len(X_train)):
    test.append((X_train[i,:],y_train[i]))
print(len(test[1][0]))
#X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#X_train = torch.Tensor(X_train) 
#y_train = torch.Tensor(y_train)



#Plot data
#x = np.linspace(0,len(train_data)-1,len(train_data))
#plt.plot(x,train_data)

