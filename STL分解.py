import pandas as pd
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
import statsmodels.api as sm

# 趋势分析
res = sm.tsa.seasonal_decompose(ts)
res.plot()
plt.show()

# 提取趋势、季节性和残差序列
trend_series = res.trend
seasonal_series = res.seasonal
residual_series = res.resid

# 存储序列
insert_index = pd.Timestamp('2013-02-01')
trend_df = pd.DataFrame({'Trend': trend_series}, index=ts.index)
data = pd.concat([data.loc[:insert_index], trend_df, data.loc[insert_index+pd.Timedelta(days=1):]], axis=1)

seasonal_df = pd.DataFrame({'Seasonal': seasonal_series}, index=ts.index)
data = pd.concat([data.loc[:insert_index], seasonal_df, data.loc[insert_index+pd.Timedelta(days=1):]], axis=1)

residual_df = pd.DataFrame({'Residual': residual_series}, index=ts.index)
data = pd.concat([data.loc[:insert_index], residual_df, data.loc[insert_index+pd.Timedelta(days=1):]], axis=1)

data.to_csv('C:/Users/86135/Desktop/作业及讲义/大二下/数模校赛/冰川/csv数据/3.1 分解结果.csv')