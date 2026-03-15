# -*- coding: utf-8 -*-
# Converted from: statistics time series/Time Series Data.ipynb

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
register_matplotlib_converters()
# 是一个 matplotlib 函数，它注册了一些自定义转换器，使得在使用 matplotlib 绘图库进行时间序列数据可视化时更加方便。
#具体来说，它可以自动将 Pandas 中的时间戳转换为 matplotlib 可以理解的格式。
#在使用 matplotlib 绘制时间序列数据时，如果不先注册这些转换器，可能会出现绘图不完整或格式错误等问题。

# %% [markdown]
# # Ice Cream Production Data

# %%
#read data
df_ice_cream = pd.read_csv('ice_cream.csv')

# %%
df_ice_cream.head()

# %%
#rename columns to something more understandable
df_ice_cream.rename(columns={'DATE':'date', 'IPN31152N':'production'}, inplace=True)

# %%
print(df_ice_cream)

# %%
#convert date column to datetime type
df_ice_cream['date'] = pd.to_datetime(df_ice_cream.date)

# %%
#set date as index
df_ice_cream.set_index('date', inplace=True)

# %%
print(df_ice_cream)

# %%
#just get data from 2010 onwards
start_date = pd.to_datetime('2010-01-01')
df_ice_cream = df_ice_cream[start_date:]

# %%
#show result
df_ice_cream.head()

# %%
plt.figure(figsize=(10,4))
plt.plot(df_ice_cream.production)
plt.title('Ice Cream Production over Time', fontsize=20)
plt.ylabel('Production', fontsize=16)
for year in range(2011,2021):
    plt.axvline(pd.to_datetime(str(year)+'-01-01'), color='k', linestyle='--', alpha=0.2)

# %% [markdown]
# # ACF

# %%
acf_plot = plot_acf(df_ice_cream.production, lags=100)

# %% [markdown]
# ## Based on decaying ACF, we are likely dealing with an Auto Regressive process

# %% [markdown]
# # PACF

# %%
pacf_plot = plot_pacf(df_ice_cream.production)

# %% [markdown]
# 我们应该从具有滞后1、2、3、10、13的自回归模型开始。PACF（部分自相关函数）可以帮助确定时间序列数据中需要多少滞后。在此过程中，首先通过对数据进行一阶差分、取对数等方法来移除趋势和季节性，然后可以利用PACF和ACF（自相关函数）图来确定适当的自回归和移动平均参数。PACF和ACF都是用于时间序列分析和预测的重要工具。

# %% [markdown]
# ## Based on PACF, we should start with an Auto Regressive model with lags 1, 2, 3, 10, 13

# %%
import yfinance as yf

# %%
#define the ticker symbol
tickerSymbol = 'SPY'

# %%
#get data on this ticker
tickerData = yf.Ticker(tickerSymbol)

# %%
#get the historical prices for this ticker
tickerDf = tickerData.history(period='1d', start='2015-1-1', end='2020-1-1')

# %%
tickerDf = tickerDf[['Close']]

# %%
#see your data
tickerDf.head()

# %%
plt.figure(figsize=(10,4))
plt.plot(tickerDf.Close)
plt.title('Stock Price over Time (%s)'%tickerSymbol, fontsize=20)
plt.ylabel('Price', fontsize=16)
for year in range(2015,2021):
    plt.axvline(pd.to_datetime(str(year)+'-01-01'), color='k', linestyle='--', alpha=0.2)

# %% [markdown]
# ## Stationarity: take first difference of this series

# %%
#take first difference
first_diffs = tickerDf.Close.values[1:] - tickerDf.Close.values[:-1]
first_diffs = np.concatenate([first_diffs, [0]])

# %%
#set first difference as variable in dataframe
tickerDf['FirstDifference'] = first_diffs

# %%
tickerDf.head()

# %%
plt.figure(figsize=(10,4))
plt.plot(tickerDf.FirstDifference)
plt.title('First Difference over Time (%s)'%tickerSymbol, fontsize=20)
plt.ylabel('Price Difference', fontsize=16)
for year in range(2015,2021):
    plt.axvline(pd.to_datetime(str(year)+'-01-01'), color='k', linestyle='--', alpha=0.2)

# %%


