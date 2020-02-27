# Simple LSTM model for stock prediction
Importing the needed libraries
```python
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

```

Downloading the available stock data from Github
```python
csv_path = 'https://raw.githubusercontent.com/Xiar-fatah/Stock-Market-Patterns/master/Tensorflow/fin.csv'
df = pd.read_csv(csv_path)
```
The data that will be pulled needs to be reshaped. The input dimension according to Keras is (batch_size, timesteps, input_dim).
batch_size will be equal to the window size, the number of days in before the prediction. timesteps is
the number of time steps that defines the number of input variables used to predict the next time step. In this case
the input variables is the data and labels are the next time step. input_dim is the number of features. The result
will be clearer with an example later on.
```python
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
```
The prediction is based on the closing price. Hence the closing price is pulled from the total
dataset. The available dataset starts at the 2020, but it is optimal to feed the machine the beginning
of the dataset, hence it is flipped with the help of numpy. The data is then normalized for optimal
computation.

```python
uni_data = df['4. close']
uni_data = np.flip(uni_data.tolist())
mean, std = uni_data.mean(),uni_data.std()
uni_data = (uni_data-mean)/std
```
The number of days in before the input is set to 20 and prediction = 0 implies that the 21 day is predicted.

```python
window = 20
prediction = 0
TRAIN_SPLIT = 4000

x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
                                           window,
                                           prediction)

x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                       window,
                                       prediction)

```



```python
BATCH_SIZE = 256
BUFFER_SIZE = 10000

train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

```

















































































