####PART1###
#IMPORTING LIBRARIES AND DATASET

import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame
import numpy as np
import math
import matplotlib.pyplot as plt  
import seaborn as seabornInstance

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from sklearn import preprocessing, svm

from sklearn.model_selection import train_test_split 
from sklearn import metrics

from matplotlib import style


start = datetime.datetime(1995, 7, 10)
end = datetime.datetime(2019, 8, 30)

df = web.DataReader("RELIANCE.NS", 'yahoo', start, end)
df.tail()

####PART2###
#CREATING MAIN DATAFRAME

dfreg = df.loc[:,['Adj Close','Volume']]
dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0
dfreg.tail()

####PART3###
#DATA WRANGLING

# Drop missing value
dfreg.fillna(value = -99999, inplace=True)
# We want to separate 1 percent of the data to forecast
forecast_out = int(math.ceil(0.01 * len(dfreg)))
# Separating the label here, we want to predict the AdjClose
forecast_col = 'Adj Close'
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
X = np.array(dfreg.drop(['label'], 1))
# Scale the X so that everyone can have the same distribution for linear regression
X = preprocessing.scale(X)
# Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
# Separate label and identify it as y
y = np.array(dfreg['label'])
y = y[:-forecast_out]
dfreg.tail()

###PART4###
#SPLITTING DATA INTO TRAINING AND TESTING
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

###PART5###
#TRAINING ML ALGORITHM

# Linear regression
clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(X_train, y_train)
# Quadratic Regression 2
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(X_train, y_train)

# Quadratic Regression 3
clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
clfpoly3.fit(X_train, y_train)

# KNN Regression
clfknn = KNeighborsRegressor(n_neighbors=2)
clfknn.fit(X_train, y_train)


###PART6###
#PERFORMANCE TESTING OF THE TRAINED ALGORITHM

confidencereg = clfreg.score(X_test, y_test)
confidencepoly2 = clfpoly2.score(X_test,y_test)
confidencepoly3 = clfpoly3.score(X_test,y_test)
confidenceknn = clfknn.score(X_test, y_test)

print('The linear regression confidence is ',confidencereg)
print('The quadratic regression 2 is ',confidencepoly2)
print('The quadratic regression 3 is ',confidencepoly3)
print('The knn regression confidence is ',confidenceknn)

###PART7###
#FORCASTING USING THE ALGORITHM HAVING BEST SCORE

# Printing the forecast
forecast_set = clfpoly3.predict(X_lately)
dfreg['Forecast'] = np.nan
print(forecast_set, confidencepoly3, forecast_out)


###PART8###
#VISUALIZING THE DATA

last_date = dfreg.iloc[-1].name
last_unix = last_date
next_unix = last_unix + datetime.timedelta(days=1)

for i in forecast_set:
    next_date = next_unix
    next_unix += datetime.timedelta(days=1)
    dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]
dfreg['Adj Close'].tail(500).plot()
dfreg['Forecast'].tail(500).plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()