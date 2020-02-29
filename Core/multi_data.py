import numpy as np
import pandas as pd
import torch 

csv_path = 'https://raw.githubusercontent.com/Xiar-fatah/Stock-Market-Patterns/master/Multiple_Features/fin.csv'
df = pd.read_csv(csv_path)

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(dataset[indices])

    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])

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
                                              shuffle=False, num_workers=0)

x_test, y_test= torch.Tensor(x_test), torch.Tensor(y_test)
test = torch.utils.data.TensorDataset(x_test,y_test)

testloader = torch.utils.data.DataLoader(test , batch_size=1,
                                              shuffle=False, num_workers=0)




