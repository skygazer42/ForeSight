# -*- coding: utf-8 -*-
# Converted from: statistics time series/MA Model.ipynb

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from datetime import datetime, timedelta
register_matplotlib_converters()

# %% [markdown]
# # Generate Some Data

# %% [markdown]
# # $y_t = 50 + 0.4\varepsilon_{t-1} + 0.3\varepsilon_{t-2} + \varepsilon_t$
# # $\varepsilon_t \sim N(0,1)$

# %%
errors = np.random.normal(0, 1, 400)

# %%
date_index = pd.date_range(start='9/1/2019', end='1/1/2020')

# %%
mu = 50
series = []
for t in range(1,len(date_index)+1):
    series.append(mu + 0.4*errors[t-1] + 0.3*errors[t-2] + errors[t])

# %%
series = pd.Series(series, date_index)
series = series.asfreq(pd.infer_freq(series.index))

# %%
plt.figure(figsize=(10,4))
plt.plot(series)
plt.axhline(mu, linestyle='--', color='grey')

# %%
def calc_corr(series, lag):
    return pearsonr(series[:-lag], series[lag:])[0]

# %% [markdown]
# # ACF

# %%
acf_vals = acf(series)
num_lags = 10
plt.bar(range(num_lags), acf_vals[:num_lags])

# %% [markdown]
# # PACF

# %%
pacf_vals = pacf(series)
num_lags = 25
plt.bar(range(num_lags), pacf_vals[:num_lags])

# %% [markdown]
# # Get training and testing sets

# %%
train_end = datetime(2019,12,30)
test_end = datetime(2020,1,1)

train_data = series[:train_end]
test_data = series[train_end + timedelta(days=1):test_end]

# %% [markdown]
# # Fit ARIMA Model

# %%
#create the model
model = ARIMA(train_data, order=(0,0,2))

# %%
#fit the model
model_fit = model.fit()

# %%
#summary of the model
print(model_fit.summary())

# %% [markdown]
# # Predicted Model:
# # $\hat{y}_t = 50 + 0.37\varepsilon_{t-1} + 0.25\varepsilon_{t-2}$

# %%
#get prediction start and end dates
pred_start_date = test_data.index[0]
pred_end_date = test_data.index[-1]

# %%
#get the predictions and residuals
predictions = model_fit.predict(start=pred_start_date, end=pred_end_date)

# %%
residuals = test_data - predictions

# %%
plt.figure(figsize=(10,4))

plt.plot(series[-14:])
plt.plot(predictions)

plt.legend(('Data', 'Predictions'), fontsize=16)

# %%
print('Mean Absolute Percent Error:', round(np.mean(abs(residuals/test_data)),4))

# %%
print('Root Mean Squared Error:', np.sqrt(np.mean(residuals**2)))

# %%


