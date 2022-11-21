import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df=pd.read_csv('sondata.csv')
print(df.shape)
df.shape

x=df['unix']
y=df['Volume USD']

plt.plot(x, y, color='royalblue', lw=2)
plt.title("Bitcoin Price over time", size=25)
plt.xlabel("Time", size=20)
plt.ylabel("$ Price", size=20)
plt.show()
df['date']=pd.to_datetime(df['date'])

df.dropna(inplace=True)
gerekliveriler = ['open', 'high', 'low','Volume BTC', 'Volume USD']
output_label = 'close'
x_train, x_test, y_train, y_test = train_test_split(df[gerekliveriler],df[output_label],test_size = 0.3)
model = LinearRegression()
model.fit(x_train, y_train)
print(model.score(x_test, y_test)*100)
future_set = df.shift(periods=30)
future_set.dropna(inplace=True)
prediction = model.predict(future_set[gerekliveriler])

plt.figure(figsize = (12, 7))
plt.plot(df["date"],df["close"], color='green', lw=2)
plt.plot(future_set["date"], prediction, color='deeppink', lw=2)
plt.title("Bitcoin Price over time", size=25)
plt.xlabel("Time", size=20)
plt.ylabel("$ Price", size=20)
plt.show()