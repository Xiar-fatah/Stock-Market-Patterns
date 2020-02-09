import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing data
url = 'https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv'
read_in_dataset= pd.read_csv(url)
#Creating training dataset by extracting opening values of data
dataset = read_in_dataset.iloc[:, 1:2].values

#Machine learning models adapt better to values between 0 and 1
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
dataset_scaled = sc.fit_transform(dataset)
#Create train set
"""

"""
X_seq = [] #Samples
y_labels = [] #Labels
"""
   In X_train each column will contain a seq of 60 elements 0-59, this will be
   the opening price for 60 days and will predict the 61:th opening value which
   will be put inside y_train. Next seq will move one day and predict the 62:th
   opening value. X_train will contain 1975 elements which is 60 less than dataset
   this is due to the first 60 elements cannot be predict hence they are used for
   prediction.
"""

for i in range(60, len(dataset_scaled)): #len(dataset_scaled) = 2035
    X_seq.append(dataset_scaled[i-60:i, 0])
    
    y_labels.append(dataset_scaled[i, 0])

#To apply the data on our model it needs to be a tensor.
X_seq_tensor, y_labels_tensor = torch.Tensor(X_seq), torch.Tensor(y_labels)
"""
    The model input will be a tensor with the following form
    tensor[[the 60 elements for the 60 days of opening values, called seq],[the predicted value, also called label]]
    However, we wanna predict 100 values. Therefore we remove 100 elements.
"""
train_data = []
for i in range(len(X_seq_tensor)-100):
    train_data.append((X_seq_tensor[i,:],y_labels_tensor[i]))
    
test_data = []
for i in range(len(X_seq_tensor)-100, len(X_seq_tensor)):
    test_data.append((X_seq_tensor[i,:],y_labels_tensor[i]))
