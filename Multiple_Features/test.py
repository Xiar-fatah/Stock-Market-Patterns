import pandas as pd
import os
path = 'MSFT.xlsx'
df = pd.read_excel(r'MSFT.xlsx')

df['Ratio'] = df['Open']/df['High']
df.head(5)

