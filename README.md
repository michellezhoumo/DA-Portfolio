# Weather related energy consumption forecast

This project uncovers the relationship between weather conditions and energy consumption through novel machine learning approaches using empirical data. 

## Tools to be Used

Starting with using K-means to cluster the types of weather conditions as a way to reduce the regressor dimension in the subsequent models, 
the preprocessed electricity usage data are then trained using ARIMA and LSTM.

### K-means

To illustrate the concept of kmeans, I wrote the model using raw data from kmeans.csv

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

```
Compute the euclidean distance from each data point to centroid mu

```python
def compute_Distance(mu,x):
# Compute euclidean distance from x_i to mu
    D = []
    for i in range(len(mu)):
        D_i = []  
        for x_i in x:
            euc_Distance = sum([pow((x_ik - mu_ik),2) for x_ik, mu_ik in zip(x_i, mu[i])])
            D_i.append(euc_Distance)
        D.append(D_i)
    #print(D)
    return D
```
k-means algorithm function

```python
def kmeans(mu, x):
    # Compute euclidean distance from x_i to mu
    D = compute_Distance(mu,x)
    # Find the minimum euclidean distance, from x_i to mu
    # Figure out weather this distance is from mu_0 or mu_1
    i = 0
    Min_D = D[i]
    while i < (len(mu)-1):
        Min_D = np.minimum(Min_D,D[i+1])
        i += 1

    # Obtain clusters of C_k of x_i
    # Color the points accordingly
    C=[]
    for k in range(len(mu)):
        C_k = []
        for i in range(len(D[k])):
            if D[k][i] == Min_D[i]:
                C_k.append(x[i])
        C.append(C_k)
    return C

```

## 
