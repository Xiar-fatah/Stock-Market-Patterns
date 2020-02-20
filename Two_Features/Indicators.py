import tulipy as ti
import numpy as np
import pandas as pd
# Importing data
url = 'https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv'
read_in_dataset= pd.read_csv(url)

# The open, high .. etc needs to be numpy arrays for tulipy
data_open = np.asarray(read_in_dataset.iloc[:,1].tolist())
data_high = np.asarray(read_in_dataset.iloc[:,2].tolist())
data_low = np.asarray(read_in_dataset.iloc[:,3].tolist())
data_closing = np.asarray(read_in_dataset.iloc[:,4].tolist())

# Simple moving average for periods 5, 10 and 20.
ma5,ma10,ma20 = ti.sma(data_open,5), ti.sma(data_open,10), ti.sma(data_open,20)

# diff = EMA12-EMA26, also known as the MACD indicator
diff = ti.ema(data_closing, 12) - ti.ema(data_closing,26)

# RSI, Relative Strong Index for periods 6 and 12
rsi6 = ti.rsi(data_closing,6)
rsi12= ti.rsi(data_closing,12)

# willr, Williams Index for periods 5 and 10
willr_5,willr_10 = ti.willr(data_high,data_low,data_closing,5), ti.willr(data_high,data_low,data_closing,10)

# mom, Momentum for periods 6 and 12
mom_6,mom_12 = ti.mom(data_closing,6),ti.mom(data_closing,12)

# Price rate of change, double check the period.
roc = ti.roc(data_closing, 60)

#def print_info(indicator):
#    print("Type:", indicator.type)
#    print("Full Name:", indicator.full_name)
#    print("Inputs:", indicator.inputs)
#    print("Options:", indicator.options)
#    print("Outputs:", indicator.outputs)
#    
#print_info(ti.rsi)