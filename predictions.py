import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import matplotlib.dates as mdates
import datetime as dt
import yfinance as yf
from prophet import Prophet
def prediction():

    data=pd.read_csv('upgrade.csv')
    data=data[['Date','Price']]
    data=data.rename({'Date':'ds','Price':'y'},axis='columns')
    data['ds']=pd.to_datetime(data['ds'])
    X=data['ds']
    Y=data['y']
    data['y']=data['y'].apply(lambda x:float(x.split()[0].replace(',','')))

    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=42)
    print(Y_train)
    print(Y_test)
    m=Prophet(yearly_seasonality=True)
    m.fit(data)
    future = m.make_future_dataframe(periods=7)
    future.tail()
    forecast = m.predict(future)
    m.plot(forecast)
    plt.show()
    fig2 = m.plot_components(forecast)
    plt.show()
prediction()




