from indic import Indicators
url = 'https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv'
store = Indicators(url,10)
store_1 = store.train
print(store_1)
