# -*- coding: utf-8 -*-
# Converted from: statistics time series/ARMA Model.ipynb

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from pandas.plotting import register_matplotlib_converters
#用于计算时间序列数据的自相关函数（ACF）和偏自相关函数（PACF）
#ACF 是时间序列数据自身的自相关性，即同一时间序列在不同时间点上的相关性，可以帮助识别时间序列中的季节性和周期性变化。
#PACF 则是去除了其它影响后（"其它影响" 指的是当前自变量以外的其它自变量在模型中的影响，也就是多重共线性的影响）的纯自相关性，
#可以帮助识别时间序列中的趋势和长期相关性。
from statsmodels.tsa.stattools import acf, pacf   
from statsmodels.tsa.arima_model import ARMA,ARIMA
register_matplotlib_converters()
from time import time
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# # Catfish Sales Data

# %%
def parser(s):
    return datetime.strptime(s, '%Y-%m-%d')

# %%
#read data
catfish_sales = pd.read_csv('catfish.csv', parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
catfish_sales 

# %%
#infer the frequency of the data
catfish_sales = catfish_sales.asfreq(pd.infer_freq(catfish_sales.index))
catfish_sales

# %%
start_date = datetime(2000,1,1)
end_date = datetime(2004,1,1)
lim_catfish_sales = catfish_sales[start_date:end_date]
lim_catfish_sales 

# %%
plt.figure(figsize=(10,4))
plt.plot(lim_catfish_sales)
plt.title('Catfish Sales in 1000s of Pounds', fontsize=20)
plt.ylabel('Sales', fontsize=16)
for year in range(start_date.year,end_date.year):
    plt.axvline(pd.to_datetime(str(year)+'-01-01'), color='k', linestyle='--', alpha=0.2)
plt.axhline(lim_catfish_sales.mean(), color='r', alpha=0.2, linestyle='--')

# %%
first_diff = lim_catfish_sales.diff()[1:]

# %%
first_diff 

# %%
plt.figure(figsize=(10,4))
plt.plot(first_diff)
plt.title('First Difference of Catfish Sales', fontsize=20)
plt.ylabel('Sales', fontsize=16)
for year in range(start_date.year,end_date.year):
    plt.axvline(pd.to_datetime(str(year)+'-01-01'), color='k', linestyle='--', alpha=0.2)
plt.axhline(first_diff.mean(), color='r', alpha=0.2, linestyle='--')

# %% [markdown]
# # ACF

# %%
first_diff # 差分

# %%
acf_vals = acf(first_diff)
plt.bar(range(2), acf_vals[:2])

# %%
plt.bar(range(14), acf_vals[:14]) # 右移14月

# %% [markdown]
# ## Based on ACF, we should start with a MA(1) process

# %% [markdown]
# # PACF

# %%
pacf_vals = pacf(first_diff)
plt.bar(range(14), pacf_vals[:14])

# %% [markdown]
# ## Based on PACF, we should start with a AR(4) process

# %% [markdown]
# # Get training and testing sets

# %%
train_end = datetime(2003,7,1)
test_end = datetime(2004,1,1)

train_data = first_diff[:train_end]
test_data = first_diff[train_end + timedelta(days=1):test_end]

# %%
train_data

# %%
test_data 

# %% [markdown]
# # Fit the ARMA Model

# %%
from statsmodels.tsa.arima.model import ARIMA

# %%
# define model 
#参数包含三个数字，分别表示 AR、差分和 MA 模型的阶数
#p 表示 AR 模型中采用的滞后项数（lag order），d 表示差分次数，q 表示 MA 模型中采用的滞后项数。
model = ARIMA(train_data, order=(4,1,8))

# %%
#包含了该模型的各种信息，如拟合结果、残差序列等等。可以使用该对象的方法和属性来进一步分析该模型，如使用 forecast() 方法进行未来值的预测。

# %%
#fit the model
start = time()
model_fit = model.fit()
end = time()
print('Model Fitting Time:', end - start)

# %%
#summary of the model
print(model_fit.summary())

# %% [markdown]
# ## So the ARMA(4,1) model is:
#
# ## $\hat{y_t} = -0.87y_{t-1} - 0.42y_{t-2} - 0.56y_{t-3} - 0.61y_{t-4} + 0.52\varepsilon_{t-1}$

# %% [markdown]
# 这个模型表示在时间点 $t$，预测值 $\hat{y_t}$ 是过去四个观测值（$y_{t-1}$、$y_{t-2}$、$y_{t-3}$ 和 $y_{t-4}$）以及滞后1期的误差项（$\varepsilon_{t-1}$）的线性组合。每个观测值和误差项都乘以对应的系数（$-0.87$、$-0.42$、$-0.56$、$-0.61$ 和 $0.52$）。这些系数表示每个滞后观测值和误差项对于预测时间点 $t$ 的影响程度

# %%
test_data

# %%
test_data.index[0]

# %%
test_data.index[-1]

# %%
#get prediction start and end dates
pred_start_date = test_data.index[0]
pred_end_date = test_data.index[-1]

# %%
#get the predictions and residuals
predictions = model_fit.predict(start=pred_start_date, end=pred_end_date)
residuals = test_data - predictions

# %%
plt.figure(figsize=(10,4))
plt.plot(residuals)
plt.title('Residuals from AR Model', fontsize=20)
plt.ylabel('Error', fontsize=16)
plt.axhline(0, color='r', linestyle='--', alpha=0.2)

# %%
plt.figure(figsize=(10,4))

plt.plot(test_data)
plt.plot(predictions)

plt.legend(('Data', 'Predictions'), fontsize=16)

plt.title('First Difference of Catfish Sales', fontsize=20)
plt.ylabel('Sales', fontsize=16)

# %%
print('Root Mean Squared Error:', np.sqrt(np.mean(residuals**2)))

# %%


