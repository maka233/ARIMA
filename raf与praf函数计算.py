import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm

# 设置随机种子以确保结果可重复
np.random.seed(0)

# 生成AR(1)时间序列数据
ar = np.array([1, -0.5])  # 我们将使用的AR模型的参数
ma = np.array([1])  # 这是MA模型的参数，在这个例子中我们不需要它
n = int(1000)  # 我们将生成的数据点的数量

arma_process = sm.tsa.ArmaProcess(ar, ma)
y = arma_process.generate_sample(nsample=n)

# 绘制ACF图像
plot_acf(y, lags=20)
plt.title('ACF of AR(1) Time Series')
plt.show()

# 绘制PACF图像
plot_pacf(y, lags=20)
plt.title('PACF of AR(1) Time Series')
plt.show()