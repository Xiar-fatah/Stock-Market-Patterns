#from IPython import get_ipython
#get_ipython().magic('reset -sf')

from plotly.offline import plot
import plotly.graph_objs as go
import numpy as np
import pandas as pd
# Importing data
url = 'https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv'
read_in_dataset= pd.read_csv(url)
# Creating dataset by extracting opening values and dates of data
data_open = read_in_dataset.iloc[:,1]
data_date = read_in_dataset.iloc[:,0]
# Transform the extracted values to a list, the first values of the list are the recent ones
data_open = data_open.tolist()
data_date = data_date.tolist()

################### Test data ###################

data_open_test = data_open[0:100]
data_date_test = data_date[0:100]

fig = go.Figure()
fig.add_trace(go.Scatter(
                x = data_date,
                y = data_open,
                name="Actaul Value",
                line_color='deepskyblue',
                opacity=0.8))

fig.add_trace(go.Scatter(
                x = data_date_test,
                y = data_open_test,
                name="Predicted Value",
                line_color='green',
                opacity=0.8))

fig.update_layout(title_text='Tata Global Beverages Stock Prediction',
                  xaxis_rangeslider_visible=True,
                  xaxis_title='Time',
                  yaxis_title='US Dollars')

plot(fig)
