import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.var_model import VAR
from math import sqrt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import grangercausalitytests, adfuller

# 读取数据
df = pd.read_csv("./pollution.csv")

# 预处理数据
# 修改日期类型
df['date'] = pd.to_datetime(df.date, format='%Y/%m/%d %H:%M:%S')

# 建立日期索引
data = df.drop(['date'], axis=1)
data.index = df.date
print("初始数据：", data)

# 创建训练集与验证集
train = data[:int(0.8 * (len(data)))]
valid = data[int(0.8 * (len(data))):]
print("train:", train.shape)
print("valid:", valid.shape)


# test = data[:int(0.01 * (len(data)))]
# # 检查相关性
# fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
# test.plot(ax=ax1)  # series plot
# pd.plotting.lag_plot(test)  # 1ag plot
# plt. show()

# ADF检验
def adf(time_series):
    result = adfuller(time_series.values)
    print(' ADF statistic:%f' % result[0])
    print(' p-value:%f' % result[1])
    print(' Critical Values:')
    for key, value in result[4].items():
        print('\t%s:%.3f' % (key, value))


# print('Augmented Dickey-Fuller Test:dew Time Series')
# adf(train['dew'])
# print('Augmented Dickey-Fuller Test:temp Time Series')
# adf(train['temp'])
# print('Augmented Dickey-Fuller Test:press Time Series')
# adf(train['press'])
# print('Augmented Dickey-Fuller Test:wnd_dir Time Series')
# adf(train['wnd_dir'])
# print('Augmented Dickey-Fuller Test:wnd_spd Time Series')
# adf(train['wnd_spd'])
# print('Augmented Dickey-Fuller Test:snow Time Series')
# adf(train['snow'])
# print('Augmented Dickey-Fuller Test:rain Time Series')
# adf(train['rain'])
# print('Augmented Dickey-Fuller Test:pollution Time Series')
# adf(train['pollution'])

# 一阶差分
train_diff = train.diff().dropna()
print(train_diff)
# train_diff.plot(figsize=(10, 6))
# plt.show()

# # 检验平稳性
# print('Augmented Dickey-Fuller Test:dew Time Series')
# adf(train_diff['dew'])
# print('Augmented Dickey-Fuller Test:temp Time Series')
# adf(train_diff['temp'])
# print('Augmented Dickey-Fuller Test:press Time Series')
# adf(train_diff['press'])
# print('Augmented Dickey-Fuller Test:wnd_dir Time Series')
# adf(train_diff['wnd_dir'])
# print('Augmented Dickey-Fuller Test:wnd_spd Time Series')
# adf(train_diff['wnd_spd'])
# print('Augmented Dickey-Fuller Test:snow Time Series')
# adf(train_diff['snow'])
# print('Augmented Dickey-Fuller Test:rain Time Series')
# adf(train_diff['rain'])
# print('Augmented Dickey-Fuller Test:pollution Time Series')
# adf(train_diff['pollution'])

# print(grangercausalitytests(train_diff[['dew', 'pollution']],
#                             maxlag=15, addconst=True, verbose=True))
# print(grangercausalitytests(train_diff[['temp', 'pollution']],
#                             maxlag=15, addconst=True, verbose=True))
# print(grangercausalitytests(train_diff[['press', 'pollution']],
#                             maxlag=15, addconst=True, verbose=True))
# print(grangercausalitytests(train_diff[['wnd_dir', 'pollution']],
#                             maxlag=15, addconst=True, verbose=True))
# print(grangercausalitytests(train_diff[['wnd_spd', 'pollution']],
#                             maxlag=15, addconst=True, verbose=True))
# print(grangercausalitytests(train_diff[['snow', 'pollution']],
#                             maxlag=15, addconst=True, verbose=True))
print(grangercausalitytests(train_diff[['rain', 'pollution']],
                            maxlag=15))
