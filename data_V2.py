import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

#Importing data
url = 'https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv'
dataset= pd.read_csv(url)
#Creating training dataset by extracting opening values of data
train_data_0 = dataset.iloc[:, 1:2].values

#Machine learning models adapt better to values between 0 and 1
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
train_data_scaled = sc.fit_transform(train_data_0)
