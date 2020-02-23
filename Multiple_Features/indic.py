import tulipy as ti
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
class Indicators:
    def __init__(self, csv_data, period):
        self.open, self.close, self.high, self.low = self.get_base(csv_data)
        self.ma5, self.ma10, self.ma20 = ti.sma(self.open, 5), ti.sma(
            self.open, 10), ti.sma(self.open, 20)
        self.diff = ti.ema(self.close, 12) - ti.ema(self.close, 26)
        self.roc = ti.roc(self.close, 60)
        self.tr = ti.tr(self.high, self.low, self.close)
        self.mom6, self.mom12 = ti.mom(
            self.close, 6), ti.mom(self.close, 12)
        self.willr5, self.willr10 = ti.willr(self.high, self.low, self.close, 5), ti.willr(
            self.high, self.low, self.close, 10)
        n, m, p = 14, 14, 14
        self.k, self.d = ti.stoch(self.high, self.low, self.close, n, m, p)
        self.osc6, self.osc12 = ti.ultosc(self.high, self.low, self.close, n, m, p), ti.ultosc(
            self.high, self.low, self.close, n, m, p)
        self.rsi6, self.rsi12 = ti.rsi(self.close, 6), ti.rsi(self.close, 12)
        self.period = period
        self.train = self.train_data()
        self.test = self.test_data()
        
        
        
    def get_base(self, csv_data):
        data = pd.read_csv(csv_data)
        data_open = np.asarray(data.iloc[:, 1].tolist())
        data_high = np.asarray(data.iloc[:, 2].tolist())
        data_low = np.asarray(data.iloc[:, 3].tolist())
        data_closing = np.asarray(data.iloc[:, 4].tolist())

        return data_open, data_high, data_low, data_closing

    def train_data(self):
        sc = MinMaxScaler(feature_range = (0, 1))
        sc_open,sc_high, sc_low, sc_close = sc.fit_transform(self.open.reshape(-1,1)), sc.fit_transform(self.high.reshape(-1,1)), sc.fit_transform(self.low.reshape(-1,1)), sc.fit_transform(self.close.reshape(-1,1))
        sc_open,sc_high, sc_low, sc_close = torch.tensor(sc_open),torch.tensor(sc_high),torch.tensor(sc_low),torch.tensor(sc_close)
        train = []
        for i in range(0, 2000): 
            concate = torch.stack((sc_open[i:i + self.period], sc_high[i:i + self.period],
                                  sc_low[i:i + self.period], sc_close[i:i + self.period]), 0)
            train.append((concate, sc_close[i + self.period]))
            
        return train
    
    
    
    
    
    def test_data(self):
        return 0
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
