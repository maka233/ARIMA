import numpy as np
import pandas as pd
import statsmodels.api as sm

# 假设你的时间序列为ts，p、q、d分别为ARIMA的超参数
# 创建ARIMA模型对象
arima_model = sm.tsa.ARIMA(ts, order=(p, d, q))

# 拟合ARIMA模型
arima_result = arima_model.fit()

# 计算AIC和BIC
aic = arima_result.aic
bic = arima_result.bic

print("AIC:", aic)
print("BIC:", bic)