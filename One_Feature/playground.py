from plotly.offline import plot
import plotly.graph_objs as go
import numpy as np
import pandas as pd

#Importing data
url = 'https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv'
read_in_dataset= pd.read_csv(url)
#Creating training dataset by extracting opening values of data
dataset = read_in_dataset.iloc[:, 1:2].values
data_set_time = read_in_dataset.iloc[:, 0:1].values
################### Data ###################

data = dataset.flatten() #Flattens the numpy array to an array so plotly can read it
data_set_time = data_set_time.flatten()
test = []
for i in range(0,2035):
    test.append(data_set_time[i])
################### Prediction ###################
x_axis_p = test[0:100]

y_axis_p = data[0:100]
print(x_axis_p)

fig = go.Figure()
fig.add_trace(go.Scatter(
                x = test,
                y = data,
                name="Actaul Value",
                line_color='deepskyblue',
                opacity=0.8))

fig.add_trace(go.Scatter(
                x = x_axis_p,
                y = y_axis_p,
                name="Predicted Value",
                line_color='green',
                opacity=0.8))

fig.update_layout(title_text='Tata Global Beverages Stock Prediction',
                  xaxis_rangeslider_visible=True,
                  xaxis_title='Time',
                  yaxis_title='US Dollars')

plot(fig)
