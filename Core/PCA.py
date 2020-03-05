import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

csv_path = 'https://raw.githubusercontent.com/Xiar-fatah/Stock-Market-Patterns/master/Core/MSFT.csv'
df = pd.read_csv(csv_path).iloc[::-1] 
df = df.drop('date', 1) # Remove date
df = df.drop('5. volume', 1) # Remove volume
pca = PCA()
pca.fit(df)
df = pd.DataFrame(pca.components_)