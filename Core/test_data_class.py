import data_class
import numpy as np
csv = 'https://raw.githubusercontent.com/Xiar-fatah/Stock-Market-Patterns/master/Core/MSFT.csv'

fetch = data_class.data(train = 4000, test = 980, window = 20, csv_path = csv)


def roll(start, end, window):
    data = []
    labels = []
    df = fetch.data
    df = df.iloc[::-1]
    start = start + window
    for i in range(start, end):
        temp = []
        for col in range(0, len(df.columns)): # (5004,16)
            if (df.columns[col] == 'date') == False | (df.columns[col] == '5. volume') == False:
                col_arr = df.iloc[:,col].tolist()    
                temp.append(np.reshape(col_arr[i-window:i], window))
        data.append(np.transpose(temp))
                
    for i in range(start, end): 
        close_arr = df['4. close'].tolist()
        labels.append(close_arr[i-window:i])

    return np.array(data), np.array(labels)
train = 4000
test = 980
window = 20
test = fetch.data

# Remember to flip dataframe
x_train,y_train = roll(0, 4000, 20)
x_test, y_test = roll(0,980,20)
df = fetch.data
df = df.iloc[::-1]