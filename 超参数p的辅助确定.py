import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 参数
ar = np.array([1, -0.5, -0.4])
ma = np.array([1])

# 生成AR(2)过程
ar2_process = ArmaProcess(ar, ma)
ar2_sample = ar2_process.generate_sample(nsample=1000)

# 绘制ACF和PACF
plt.figure(figsize=(12,8))
plt.subplot(211)
plot_acf(ar2_sample, ax=plt.gca())
plt.subplot(212)
plot_pacf(ar2_sample, ax=plt.gca())
plt.show()