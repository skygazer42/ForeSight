# -*- coding: utf-8 -*-
# Converted from: ml time series/prophet.ipynb

# %% [markdown]
# # 0. 环境准备

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# 设置pandas可以显示的行数和列数
pd.options.display.max_rows = 400
pd.options.display.max_columns = None

# 忽略warnings
import warnings
warnings.filterwarnings("ignore")

#推荐安装插件： nbextensions

# %% [markdown]
# # 1.读入数据

# %%
# store:门店编号
# dept: 商品部门编号
# week: 每周周一的日期 
# sales: 销售金额
df_sales = pd.read_csv('data/store_sales.csv', parse_dates=['week'])
df_sales.head(2)

# %%
# store:门店编号
# dept: 商品部门编号
# week: 每周周一的日期 
# promotion_sales: 促销活动带来的销售金额
df_promotion = pd.read_csv('data/promotion_data.csv', parse_dates=['week'])
df_promotion.head(2)

# %%
# 合并销售和促销数据
df_all = pd.merge(
    df_sales,
    df_promotion,
    how='left',
    on=['store', 'dept', 'week'],
    validate='one_to_one',
)
df_all.fillna(0, inplace=True)

# %% [markdown]
# # 使用prophet算法做预测

# %%
import prophet

# %% [markdown]
# ## 预测1号店的1号部门， 以2012-07-30以前的数据做训练，往后预测一周

# %%
# 数据准备
df_train = df_all[ (df_all['week']<='2012-07-30') & 
                 (df_all['store']==1) & 
                 (df_all['dept']==1)]
#使用prophet前，要将日期字段重命名为”ds", 预测对象重命名为"y"
df_train.rename(columns={'week':'ds','sales':'y'},inplace=True)

df_test = df_all[(df_all['week']=='2012-08-06') &
                (df_all['store']==1) & 
                (df_all['dept']==1)]
df_test.rename(columns={'week':'ds','sales':'y'},inplace=True)

# %%
#训练模型
m = prophet.Prophet(yearly_seasonality=True)
m.add_regressor( 'promotion_sales' )
m.fit( df_train )

# %%
#  拟合历史的数据
df_fit = m.predict( df_train )

# 可视化拟合结果
fig1 = m.plot_components(df_fit)
fig2 = m.plot( df_fit )

# %%
# 预测未来的数据
df_predict = m.predict( df_test )
df_test['yhat'] = df_predict['yhat'].values

# %% [markdown]
# ## 预测1号店的所有部门， 以2012-07-30以前的数据做训练，往后预测一周

# %%
dept_list = df_all[ df_all['store']==1 ]['dept'].unique()

all_result = []
for dept in dept_list:
    # 数据准备
    df_train = df_all[ (df_all['week']<='2012-07-30') & 
                      (df_all['store']==1) & 
                      (df_all['dept']==dept)]
    df_train.rename(columns={'week':'ds','sales':'y'},inplace=True)

    df_test = df_all[ (df_all['week']>'2012-07-30') & 
                     (df_all['week']<='2012-08-06') &
                     (df_all['store']==1) & 
                     (df_all['dept']==dept)]
    df_test.rename(columns={'week':'ds','sales':'y'},inplace=True)
    
    ## 只有超过两年的历史数据才能训练模型
    if (df_train.shape[0] > 100) & ( df_test.shape[0] > 0 ):
        #训练模型
        m = prophet.Prophet(yearly_seasonality=True)
        m.add_regressor( 'promotion_sales' )
        m.fit( df_train )

        #预测结果
        df_predict = m.predict( df_test )
        df_test['yhat'] = df_predict['yhat'].values

        all_result.append( df_test )
all_result =pd.concat( all_result )

# %% [markdown]
# # 使用lightGBM算法做预测

# %%
import lightgbm as lgb

# %% [markdown]
# ## 预测1号店的1号部门， 以2012-07-30以前的数据做训练，往后预测一周

# %%
df_sample = df_all[ (df_all['store']==1) & 
                  (df_all['dept']==1)].sort_values('week')

# %%
# 特征构建
feature_cols = []

## 第一组特征： 历史数据最后一周的销量和促销活动的金额
df_sample['sales_lw'] = df_sample['sales'].shift(1)
df_sample['promotion_lw'] = df_sample['promotion_sales'].shift(1)
feature_cols = feature_cols + ['sales_lw', 'promotion_lw']

## 第二组特征： 上一个周期（即去年同一周）的销量和促销活动金额
df_sample['sales_ly'] = df_sample['sales'].shift(52)
df_sample['promotion_ly'] = df_sample['promotion_sales'].shift(52)
feature_cols = feature_cols + ['sales_ly', 'promotion_ly']

## 第三组特征：待预测周的促销活动金额
feature_cols = feature_cols + ['promotion_sales']

## 只保留所有特征都不为空的数据
for col in feature_cols:
    df_sample = df_sample[ ~df_sample[col].isna() ]

# %%
# 构建训练集和验证集
x_train = df_sample[ df_sample['week']<='2012-07-30' ][ feature_cols ].values
y_train = df_sample[ df_sample['week']<='2012-07-30' ][ 'sales' ].values

x_test = df_sample[ df_sample['week']=='2012-08-06' ][ feature_cols ].values
y_test = df_sample[ df_sample['week']=='2012-08-06'  ][ 'sales' ].values

# %%
# 使用lightGBM建模
from lightgbm.sklearn import LGBMRegressor

model = LGBMRegressor()
model.fit(x_train, y_train)

# %%
# 预测
y_pred = model.predict( x_test )

# %%
print(y_pred, y_test)

# %% [markdown]
# ## 预测1号店的所有部门， 以2012-07-30以前的数据做训练，往后预测一周

# %%
df_sample = df_all[ (df_all['store']==1)].sort_values(['dept','week'])

# %%
# 特征构建
feature_cols = []

## 第一组特征： 历史数据最后一周的销量和促销活动的金额
df_sample['sales_lw'] = df_sample.groupby(['dept'])['sales'].shift(1)
df_sample['promotion_lw'] = df_sample.groupby(['dept'])['promotion_sales'].shift(1)
feature_cols = feature_cols + ['sales_lw', 'promotion_lw']

## 第二组特征： 上一个周期（即去年同一周）的销量和促销活动金额
df_sample['sales_ly'] = df_sample.groupby(['dept'])['sales'].shift(52)
df_sample['promotion_ly'] = df_sample.groupby(['dept'])['promotion_sales'].shift(52)
feature_cols = feature_cols + ['sales_ly', 'promotion_ly']

## 第三组特征：待预测周的促销活动金额
feature_cols = feature_cols + ['promotion_sales']

## 只保留所有特征都不为空的数据
for col in feature_cols:
    df_sample = df_sample[ ~df_sample[col].isna() ]

# %%
# 构建训练集和验证集
x_train = df_sample[ df_sample['week']<='2012-07-30' ][ feature_cols ].values
y_train = df_sample[ df_sample['week']<='2012-07-30' ][ 'sales' ].values

x_test = df_sample[ df_sample['week']=='2012-08-06' ][ feature_cols ].values
y_test = df_sample[ df_sample['week']=='2012-08-06'  ][ 'sales' ].values

# %%
# 使用lightGBM建模
model = LGBMRegressor()
model.fit(x_train, y_train)

# %%
# 预测
y_pred = model.predict( x_test )

# %%
print(y_pred, y_test)

# %%


