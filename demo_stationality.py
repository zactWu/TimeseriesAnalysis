import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.var_model import VAR
from math import sqrt
from sklearn.metrics import mean_squared_error

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
# # checking the correlations between X(t) and X(t-1)
# fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
# test.plot(ax=ax1)  # series plot
# pd.plotting.lag_plot(test)  # 1ag plot
# plt. show()

train_diff = train.diff().dropna()
print(train_diff)
train_diff.plot(figsize=(10, 6))
plt.show()
