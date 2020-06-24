# Weather related energy consumption forecast

This project uncovers the relationship between weather conditions and energy consumption through novel machine learning approaches using empirical data. 

## Tools to be Used

Starting with using K-means to cluster the types of weather conditions as a way to reduce the regressor dimension in the subsequent models, 
the preprocessed electricity usage data are then trained using ARIMA and LSTM.

### K-means

Illustrate the concept of kmeans using kmeans.csv

```python
import pandas as pd
#import os

#print(os.getcwd())
df = pd.read_csv('kmeans.csv',header = None)
#print(df.head())
x = df.to_numpy().tolist()
C = x

#inital state of all xi
plt.scatter(np.array(C)[:,0],np.array(C)[:,1])

#define initial mu
mu_ini = [[1,2],[-2,-1.5]]



## 
