import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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
#reshape into an numpy array
X_train, y_train = np.array(X_train), np.array(y_train)

#Reshaping into three dimensional tensor
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Plot data
#x = np.linspace(0,len(train_data)-1,len(train_data))
#plt.plot(x,train_data)

