import pandas as pd

df_1 = pd.read_csv('FORD.csv')
df_2 = pd.read_csv('IBM.csv')
df_1= df_1.drop('Unnamed: 0', 1)
df_2= df_2.drop('Unnamed: 0', 1)
df_1.to_csv("FORD.csv")
df_2.to_csv("IBM.csv")

