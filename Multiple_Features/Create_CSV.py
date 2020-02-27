from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import pandas as pd
import time


# Your key here
key = '2A5W3P15R5XIWSPO'

# Returns and saves technicial indicators in pandas formation respectivly csv file.
class Indicators:
    def __init__(self, company):
        # Company is the stock name.
        company = company
        csv = self.get_base(company)
        
    def lowest_val(self,csv_data):
        # The length of each column without Nan values
        no_nan_len = csv_data.isnull().sum()
        lowest = 0
        for i in range(0,len(no_nan_len)):
            if lowest < no_nan_len[i]:
                lowest = no_nan_len[i]
        return lowest

    def get_base(self, company):

        ts = TimeSeries(key, output_format='pandas')
        ti = TechIndicators(key, output_format='pandas')
        
        
        # Get the data, returns a tuple
        # aapl_data is a pandas dataframe, _ is a dict
        csv_data, _ = ts.get_daily(symbol=company, outputsize = 'full')
        # Remove the day today, due to time 
        csv_data = csv_data[1:]
        csv_sma5, _ = ti.get_sma(symbol=company,time_period='5')
        csv_sma10, _ = ti.get_sma(symbol=company,time_period='10')
        csv_sma20, _ = ti.get_sma(symbol=company,time_period='20')
        csv_mom6, _ = ti.get_mom(symbol=company,time_period='6')
        time.sleep(61)
        csv_mom12, _ = ti.get_mom(symbol=company,time_period='12')
        csv_ultosc, _ = ti.get_ultosc(symbol=company)
        csv_macd, _ = ti.get_macd(symbol=company)
        csv_rsi6, _ = ti.get_rsi(symbol=company,time_period='6')
        time.sleep(61)
        csv_rsi12, _ = ti.get_rsi(symbol=company,time_period='12')
        csv_willr, _ = ti.get_willr(symbol=company)
        csv_roc, _ = ti.get_roc(symbol=company)
        csv_stoch,_ = ti.get_stoch(symbol=company)
        
        csv_data['SMA5'] = csv_sma5
        csv_data['SMA10'] = csv_sma10
        csv_data['SMA20'] = csv_sma20
        csv_data['MOM6'] = csv_mom6
        csv_data['MOM12'] = csv_mom12
        csv_data['ULTOSC'] = csv_ultosc
        csv_data['RSI6'] = csv_rsi6
        csv_data['RSI12'] = csv_rsi12
        csv_data['WILLR'] = csv_willr
        csv_data['ROC'] = csv_roc
        
        
        # TODO: FIX THESE INDICATORS
        #MSFT_data['D'] = MSFT_stoch_D
        #MSFT_data['K'] = MSFT_stoch_K
        #MSFT_data['MACD'] = MSFT_macd

        
        lowest = self.lowest_val(csv_data)
        csv_data = csv_data[:-lowest]
        csv_data.to_csv("fin.csv")
        return csv_data

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
