import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df_train = pd.read_csv('/home/gamooga/data/timeseries/daily-total-female-births.csv')

## Inital data exploration

print (df_train.describe())

print (df_train.head())

print (type(df_train))

print (df_train.size)

print (df_train['Births'].plot())
plt.show()

## Smoothening the data, using moving average

df_train_mean = df_train.rolling(window = 20).mean()

df_train_mean['Births'].plot()
plt.show()

## Using a naive baseline model

df_naive = pd.concat([df_train.Births,df_train.Births.shift(1)],axis=1)

df_naive.columns = ['Actual_births','Forecast_births']

print (df_naive.head())

from sklearn.metrics import mean_squared_error
import numpy as np

df_naive = df_naive[1:]

print (df_naive.tail())

df_error = mean_squared_error(df_naive.Actual_births,df_naive.Forecast_births)

print (np.sqrt(df_error))


## Plotting auto-correlation function (ACF) and partial auto-correlation function (PACF)

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(df_train.Births, lags = 100)
plt.show()


plot_pacf(df_train.Births, lags = 100)
plt.show()


## p value from pacf, q value from acf model


birth_train_df =df_train[1:335]
birth_test_df  =df_train[335:]

print (birth_train_df.size)
print (birth_test_df.size)

## Using ARIMA model

from statsmodels.tsa.arima_model import ARIMA

forecast_model = ARIMA(birth_train_df.Births, order=(2,1,3))

forecast_model_fit = forecast_model.fit()

print (forecast_model_fit.aic)

birth_forecast = forecast_model_fit.forecast( steps = 30)[0]

print (birth_forecast)

print (birth_test_df)

print (np.sqrt(mean_squared_error(birth_test_df.Births,birth_forecast)))
