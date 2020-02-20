import tulipy as ti
import numpy as np
import pandas as pd
# Importing data
url = 'https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv'
read_in_dataset= pd.read_csv(url)

data_open = np.asarray(read_in_dataset.iloc[:,1].tolist())
data_high = np.asarray(read_in_dataset.iloc[:,2].tolist())
data_low = np.asarray(read_in_dataset.iloc[:,3].tolist())
data_closing = np.asarray(read_in_dataset.iloc[:,4].tolist())

# Simple moving average for periods 5, 10 and 20.
ma5,ma10,ma20 = ti.sma(data_open,5), ti.sma(data_open,10), ti.sma(data_open,20)

# diff = EMA12-EMA26 
diff = ti.ema(data_closing, 12) - ti.ema(data_closing,26)

# RSI, Relative Strong Index


#def print_info(indicator):
#    print("Type:", indicator.type)
#    print("Full Name:", indicator.full_name)
#    print("Inputs:", indicator.inputs)
#    print("Options:", indicator.options)
#    print("Outputs:", indicator.outputs)
#    
#print_info(ti.rsi)