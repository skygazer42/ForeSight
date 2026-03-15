# -*- coding: utf-8 -*-
# Converted from: statistics time series/Anomaly Detection.ipynb

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from pandas.plotting import register_matplotlib_converters
#ACF 是一个完整的自相关函数，可为我们提供具有滞后值的任何序列的自相关值。
#ACF 考虑所有周期的相关性，PACF 则只考虑特定周期的相关性
#它描述了该序列的当前值与其过去的值之间的相关程度。 时间序列可以包含趋势，季节性，周期性和残差等成分。 ACF 在寻找相关性时会考虑所有这些成分。
from statsmodels.tsa.stattools import acf, pacf  
#SARIMAX 是在差分移动自回归模型（ARIMA）的基础上加上季节（S,Seasonal）和外部因素 (X,eXogenous)。也就是说以 ARIMA 基础加上周期性和季节性
from statsmodels.tsa.statespace.sarimax import SARIMAX
register_matplotlib_converters()
from time import time
import warnings
warnings.filterwarnings('ignore')

CATFISH_TITLE = 'Catfish Sales in 1000s of Pounds'

# %% [markdown]
# # Catfish Sales Data

# %%
def parser(s):
    return datetime.strptime(s, '%Y-%m-%d')

# %%
#read data
catfish_sales = pd.read_csv('catfish.csv', parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

# %%
print(catfish_sales)

# %%
#infer the frequency of the data
catfish_sales = catfish_sales.asfreq(pd.infer_freq(catfish_sales.index))

# %%
print(catfish_sales.index)

# %%
#pd.infer_freq(catfish_sales.index) 来推断出该时间序列数据的采样频率（即时间间隔）
pd.infer_freq(catfish_sales.index) #表示时间序列数据按月进行采样，其中 "MS" 是 pandas 库中表示月初（Month Start）的一个字符串别名

# %%
catfish_sales.asfreq(pd.infer_freq(catfish_sales.index)) # asfreq() 函数来按照 MS 进行重新采样

# %%
datetime(1996,1,1) #年 月 日 时 分

# %%
start_date = datetime(1996,1,1)
end_date = datetime(2000,1,1)
lim_catfish_sales = catfish_sales[start_date:end_date]

# %%
lim_catfish_sales.head(10)

# %% [markdown]
# Introduce an Anomaly

# %%
#At December 1 1998
lim_catfish_sales[datetime(1998,12,1)] = 10000

# %%
plt.figure(figsize=(10,4))
plt.plot(lim_catfish_sales)
plt.title(CATFISH_TITLE, fontsize=20)
plt.ylabel('Sales', fontsize=16)
for year in range(start_date.year,end_date.year):
    plt.axvline(pd.to_datetime(str(year)+'-01-01'), color='k', linestyle='--', alpha=0.2)

# %% [markdown]
# ## Remove the trend

# %%
lim_catfish_sales.diff().head() # 中相邻数据之间的差分

# %%
first_diff = lim_catfish_sales.diff()[1:]

# %%
first_diff.head()

# %%
plt.figure(figsize=(10,4))
plt.plot(first_diff)
plt.title(CATFISH_TITLE, fontsize=20)
plt.ylabel('Sales', fontsize=16)
for year in range(start_date.year,end_date.year):
    plt.axvline(pd.to_datetime(str(year)+'-01-01'), color='k', linestyle='--', alpha=0.2)
plt.axhline(0, color='k', linestyle='--', alpha=0.2)

# %% [markdown]
# # Get training and testing sets

# %%
train_end = datetime(1999,7,1)
test_end = datetime(2000,1,1)

test_data = lim_catfish_sales[train_end + timedelta(days=1):test_end]

# %% [markdown]
# # Make Predictions

# %%
print(test_data)

# %%
my_order = (0,1,0)
my_seasonal_order = (1, 0, 1, 12)

# %%
print(my_seasonal_order)

# %%
print(test_data.index)

# %%
print(lim_catfish_sales[: train_end - timedelta(days=1)])

# %%
#采用 SARIMA 模型，用于预测时间序列数据在未来若干时间点上的值。
rolling_predictions = test_data.copy() #先创建了一个名为 rolling_predictions 的 DataFrame 对象，用于存储预测结果。
for train_end in test_data.index:
    train_data = lim_catfish_sales[:train_end-timedelta(days=1)] #对于 test_data 中的每个时间点（即索引），从历史数据 lim_catfish_sales 中提取训练数据
    model = SARIMAX(train_data, order=my_order, seasonal_order=my_seasonal_order) #使用 SARIMA 模型对这些数据进行拟合和预测
    model_fit = model.fit()

    pred = model_fit.forecast() #预测的结果 pred 将被存储在 rolling_predictions 中相应时间点的位置上
    rolling_predictions[train_end] = pred

# %%
print(rolling_predictions)

# %%
print(rolling_predictions)

# %%
print(test_data)

# %%
rolling_residuals = test_data - rolling_predictions

# %%
plt.figure(figsize=(10,4))
plt.plot(rolling_residuals)
plt.axhline(0, linestyle='--', color='k')
plt.title('Rolling Forecast Residuals from SARIMA Model', fontsize=20)
plt.ylabel('Error', fontsize=16)

# %%
plt.figure(figsize=(10,4))

plt.plot(lim_catfish_sales)
plt.plot(rolling_predictions)

plt.legend(('Data', 'Predictions'), fontsize=16)

plt.title(CATFISH_TITLE, fontsize=20)
plt.ylabel('Production', fontsize=16)
for year in range(start_date.year,end_date.year):
    plt.axvline(pd.to_datetime(str(year)+'-01-01'), color='k', linestyle='--', alpha=0.2)

# %%
print('Mean Absolute Percent Error:', round(np.mean(abs(rolling_residuals/test_data)),4))

# %%
print('Root Mean Squared Error:', np.sqrt(np.mean(rolling_residuals**2)))

# %% [markdown]
# # Detecting the Anomaly 检测异常

# %% [markdown]
# ## Attempt 1: Deviation Method 偏差法

# %%
plt.figure(figsize=(10,4))
plt.plot(lim_catfish_sales)
plt.title(CATFISH_TITLE, fontsize=20)
plt.ylabel('Sales', fontsize=16)
for year in range(start_date.year,end_date.year):
    plt.axvline(pd.to_datetime(str(year)+'-01-01'), color='k', linestyle='--', alpha=0.2)

# %%
rolling_deviations = pd.Series(dtype=float, index = lim_catfish_sales.index)

# %%
print(rolling_deviations)

# %%
for date in rolling_deviations.index:
    #get the window ending at this data point
    window = lim_catfish_sales.loc[:date]
    
    #get the deviation within this window
    rolling_deviations.loc[date] = window.std()

# %%
#get the difference in deviation between one time point and the next
diff_rolling_deviations = rolling_deviations.diff()
diff_rolling_deviations = diff_rolling_deviations.dropna()

# %%
plt.figure(figsize=(10,4))
plt.plot(diff_rolling_deviations)
plt.title('Deviation Differences', fontsize=20)
plt.ylabel('Sales', fontsize=16)
for year in range(start_date.year,end_date.year):
    plt.axvline(pd.to_datetime(str(year)+'-01-01'), color='k', linestyle='--', alpha=0.2)

# %% [markdown]
# ## Attempt 2: Seasonal Method

# %%
month_deviations = lim_catfish_sales.groupby(lambda d: d.month).std()

# %%
plt.figure(figsize=(10,4))
plt.plot(month_deviations)
plt.title('Deviation by Month', fontsize=20)
plt.ylabel('Sales', fontsize=16)

# %% [markdown]
# ## So, the anomaly occurs in a December

# %%
december_data = lim_catfish_sales[lim_catfish_sales.index.month == 12]

# %%
print(december_data)

# %%
min_dev = 9999999
curr_anomaly = None
for date in december_data.index:
    other_data = december_data[december_data.index != date]
    curr_dev = other_data.std()
    if curr_dev < min_dev:
        min_dev = curr_dev
        curr_anomaly = date

# %%
print(curr_anomaly)

# %% [markdown]
# # What to do about the anomaly? 如何处理这个异常现象

# %% [markdown]
# ## Simple Idea: use mean of other months

# %%
adjusted_data = lim_catfish_sales.copy()
adjusted_data.loc[curr_anomaly] = december_data[(december_data.index != curr_anomaly) & (december_data.index < test_data.index[0])].mean()

# %%
plt.figure(figsize=(10,4))
plt.plot(lim_catfish_sales, color='firebrick', alpha=0.4)
plt.plot(adjusted_data)
plt.title(CATFISH_TITLE, fontsize=20)
plt.ylabel('Sales', fontsize=16)
for year in range(start_date.year,end_date.year):
    plt.axvline(pd.to_datetime(str(year)+'-01-01'), color='k', linestyle='--', alpha=0.2)
plt.axvline(curr_anomaly, color='k', alpha=0.7)

# %% [markdown]
# # Resulting Predictions

# %%
train_end = datetime(1999,7,1)
test_end = datetime(2000,1,1)

test_data = adjusted_data[train_end + timedelta(days=1):test_end]

# %%
rolling_predictions = test_data.copy()
for train_end in test_data.index:
    train_data = adjusted_data[:train_end-timedelta(days=1)]
    model = SARIMAX(train_data, order=my_order, seasonal_order=my_seasonal_order)
    model_fit = model.fit()

    pred = model_fit.forecast()
    rolling_predictions[train_end] = pred

# %%
rolling_residuals = test_data - rolling_predictions

# %%
plt.figure(figsize=(10,4))
plt.plot(rolling_residuals)
plt.axhline(0, linestyle='--', color='k')
plt.title('Rolling Forecast Residuals from SARIMA Model', fontsize=20)
plt.ylabel('Error', fontsize=16)

# %%
plt.figure(figsize=(10,4))

plt.plot(lim_catfish_sales)
plt.plot(rolling_predictions)

plt.legend(('Data', 'Predictions'), fontsize=16)

plt.title(CATFISH_TITLE, fontsize=20)
plt.ylabel('Production', fontsize=16)
for year in range(start_date.year,end_date.year):
    plt.axvline(pd.to_datetime(str(year)+'-01-01'), color='k', linestyle='--', alpha=0.2)

# %%
print('Mean Absolute Percent Error:', round(np.mean(abs(rolling_residuals/test_data)),4))

# %%
print('Root Mean Squared Error:', np.sqrt(np.mean(rolling_residuals**2)))

# %%


