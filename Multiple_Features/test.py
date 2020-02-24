from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import pandas as pd
import time


# Your key here
key = '2A5W3P15R5XIWSPO'
# Chose your output format, or default to JSON (python dict)
ts = TimeSeries(key, output_format='pandas')
ti = TechIndicators(key, output_format='pandas')

# Get the data, returns a tuple
# aapl_data is a pandas dataframe, _ is a dict
MSFT_data, _ = ts.get_daily(symbol='MSFT', outputsize = 'full')
MSFT_data = MSFT_data[1:]
# aapl_sma is a dict, aapl_meta_sma also a dict
MSFT_sma5, _ = ti.get_sma(symbol='MSFT',time_period='5')
MSFT_sma10, _ = ti.get_sma(symbol='MSFT',time_period='10')
MSFT_sma20, _ = ti.get_sma(symbol='MSFT',time_period='20')
MSFT_mom6, _ = ti.get_mom(symbol='MSFT',time_period='6')
time.sleep(61)
MSFT_mom12, _ = ti.get_mom(symbol='MSFT',time_period='12')
MSFT_ultosc, _ = ti.get_ultosc(symbol='MSFT')
#MSFT_ema, _ = ti.get_ema(symbol='MSFT')
MSFT_macd, _ = ti.get_macd(symbol='MSFT')
MSFT_rsi6, _ = ti.get_rsi(symbol='MSFT',time_period='6')
time.sleep(61)
MSFT_rsi12, _ = ti.get_rsi(symbol='MSFT',time_period='12')
MSFT_willr, _ = ti.get_willr(symbol='MSFT')
MSFT_roc, _ = ti.get_roc(symbol='MSFT')
MSFT_stoch,_ = ti.get_stoch(symbol='MSFT')

MSFT_data['SMA5'] = MSFT_sma5
MSFT_data['SMA10'] = MSFT_sma10
MSFT_data['SMA20'] = MSFT_sma20
MSFT_data['MOM6'] = MSFT_mom6
MSFT_data['MOM12'] = MSFT_mom12
MSFT_data['ULTOSC'] = MSFT_ultosc
#MSFT_data['MACD'] = MSFT_macd
MSFT_data['RSI6'] = MSFT_rsi6
MSFT_data['RSI12'] = MSFT_rsi12
MSFT_data['WILLR'] = MSFT_willr
MSFT_data['ROC'] = MSFT_roc
# TODO: Find the lowest val
MSFT_data = MSFT_data[:5004]
MSFT_data.to_csv("fin.csv")
# TODO: FIX THESE INDICATORS
#MSFT_data['D'] = MSFT_stoch_D
#MSFT_data['K'] = MSFT_stoch_K

