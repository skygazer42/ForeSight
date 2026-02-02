# -*- coding: utf-8 -*-
# Converted from: statistics time series/Time Series Data Preprocessing.ipynb

# %%
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# %% [markdown]
# # Read the Data

# %%
def parser(s):
    return datetime.strptime(s, '%Y-%m')

# %%
ice_cream_heater_df = pd.read_csv('ice_cream_vs_heater.csv', parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

# %%
ice_cream_heater_df = ice_cream_heater_df.asfreq(pd.infer_freq(ice_cream_heater_df.index))

# %%
ice_cream_heater_df

# %%
heater_series = ice_cream_heater_df.heater

# %%
heater_series

# %%
def plot_series(series):
    plt.figure(figsize=(12,6))
    plt.plot(heater_series, color='red')
    plt.ylabel('Search Frequency for "Heater"', fontsize=16)

    for year in range(2004, 2021):
        plt.axvline(datetime(year,1,1), linestyle='--', color='k', alpha=0.5)

# %%
plot_series(heater_series)

# %% [markdown]
# # Normalize

# %%
avg, dev = heater_series.mean(), heater_series.std()

# %%
heater_series = (heater_series - avg) / dev

# %%
plot_series(heater_series)
plt.axhline(0, linestyle='--', color='k', alpha=0.3)

# %% [markdown]
# # Take First Difference to Remove Trend

# %% [markdown]
# 趋势指的是时间序列中长期变化的方向和速度，通常可以用一条直线来表示。在金融领域，趋势也指股票价格、指数等长期的变化方向。

# %% [markdown]
# 取一阶差分以去除趋势  时间序列= 趋势+ 残差+ 季节性

# %%
heater_series = heater_series.diff().dropna()

# %%
plot_series(heater_series)
plt.axhline(0, linestyle='--', color='k', alpha=0.3)

# %% [markdown]
# # Remove Increasing Volatility

# %% [markdown]
# 通常时间序列分析包含四个主要成分：**趋势、季节性、循环性和随机性（或残差）**。在这个框架下，波动性（volatility）通常被视为循环性的一种形式，它描述了时间序列中波动的大小和频率。波动性通常是随机的，并且可以通过对时间序列进行分解来检查其他成分。因此，波动性可以看作是随机性中的一个方面，而随机性是由除趋势、季节性和循环性之外的因素引起的。

# %%
annual_volatility = heater_series.groupby(heater_series.index.year).std()

# %%
annual_volatility

# %%
heater_annual_vol = heater_series.index.map(lambda d: annual_volatility.loc[d.year])

# %%
heater_annual_vol

# %%
heater_series = heater_series / heater_annual_vol

# %%
plot_series(heater_series)
plt.axhline(0, linestyle='--', color='k', alpha=0.3)

# %% [markdown]
# # Remove Seasonality

# %%
month_avgs = heater_series.groupby(heater_series.index.month).mean()

# %%
month_avgs

# %%
heater_month_avg = heater_series.index.map(lambda d: month_avgs.loc[d.month])

# %%
heater_month_avg

# %%
heater_series = heater_series - heater_month_avg

# %%
plot_series(heater_series)
plt.axhline(0, linestyle='--', color='k', alpha=0.3)

