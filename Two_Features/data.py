#from IPython import get_ipython
#get_ipython().magic('reset -sf')
import torch
import numpy as np
import pandas as pd
################### Total data ###################

# Importing data
url = 'https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv'
read_in_dataset= pd.read_csv(url)
# Creating dataset by extracting opening values and dates of data
# data_open is used to transform the opening prices to tensors
data_open = read_in_dataset.iloc[:,1:2]
data_closing = read_in_dataset.iloc[:, 5:6]
# data_open_list is used to plot the data
data_open_list = read_in_dataset.iloc[:,1]

data_date = read_in_dataset.iloc[:,0]
# Transform the extracted values to a list, the first values of the list are the recent ones
data_open_list = data_open_list.tolist()
data_date = data_date.tolist()

################### Test data ###################

# The dates for the testing data
data_date_test = data_date[0:100] 

################### Tensor data ###################
# Normalizing the data to take on values between 0 and 1
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
dataset_scaled_opening = sc.fit_transform(data_open)
dataset_scaled_closing = sc.fit_transform(data_closing)
# Create train set

X_seq_open = [] #Samples
X_seq_closing = []
y_labels = [] #Labels
"""
   In X_train each column will contain a seq of 60 elements 0-59, this will be
   the opening price for 60 days and will predict the 61:th opening value which
   will be put inside y_train. Next seq will move one day and predict the 62:th
   opening value. X_train will contain 1975 elements which is 60 less than dataset
   this is due to the first 60 elements cannot be predict hence they are used for
   prediction.
"""

for i in range(60, len(dataset_scaled_opening)): #len(dataset_scaled) = 2035
    X_seq_open.append(dataset_scaled_opening[i-60:i, 0])
    X_seq_closing.append(dataset_scaled_closing[i-60:i, 0])


    
    y_labels.append(dataset_scaled_opening[i, 0])
    

# To apply the data on our model it needs to be a tensor.
X_seq_tensor_open, y_labels_tensor = torch.Tensor(X_seq_open), torch.Tensor(y_labels)
X_seq_tensor_closing = torch.Tensor(X_seq_closing)
"""
    The model input will be a tensor with the following form
    tensor[[the 60 elements for the 60 days of opening values, called seq],[the predicted value, also called label]]
    However, we wanna predict 100 values. Therefore we remove 100 elements.
"""
train_data = []
for i in range(100,len(X_seq_tensor_open)):
    # Create a matrix, the first row is the opening values, second closing values
    concate = torch.stack((X_seq_tensor_open[len(X_seq_tensor_open)-i,:], 
                                       X_seq_tensor_closing[len(X_seq_tensor_open)-i,:]), 0)
#    print(X_seq_tensor_open[len(X_seq_tensor_open)-i,:])
#    print(X_seq_tensor_closing[len(X_seq_tensor_open)-i,:])
##    raise ValueError('whops')
    train_data.append((concate, y_labels_tensor[len(X_seq_tensor_open)-i]))
t = concate.view(1,60,2)
#print(t)
test_data = []
for i in range(0, 100):
    concate = torch.stack((X_seq_tensor_open[100-i,:], 
                                   X_seq_tensor_closing[100-i,:]), 0)
    
    test_data.append((concate,y_labels_tensor[100-i]))


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    