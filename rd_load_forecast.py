import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

class LoadForecast:
    def __init__(self, historical_data):
        self.historical_data = historical_data
        self.model = None

    def train_arima(self, order=(5, 1, 0)):
        # 创建和训练 ARIMA 模型
        self.model = ARIMA(self.historical_data, order=order)
        self.model = self.model.fit()
        print(f"ARIMA model trained with order {order}")
    
    def forecast(self, steps):
        if self.model is None:
            raise ValueError("Model is not trained. Call train_arima first.")
        forecast_result = self.model.forecast(steps=steps)
        return forecast_result

# 使用随机数据进行训练
if __name__ == "__main__":
    # 生成一些示例数据
    np.random.seed(0)
    data = np.random.randn(100).cumsum()  # 累加和，模拟发电量数据
    forecast_steps = 24  # 预测24步，即24小时

    # 创建和训练发电量预测模型
    load_forecast = LoadForecast(data)
    load_forecast.train_arima(order=(5, 1, 0))

    # 生成预测
    prediction = load_forecast.forecast(steps=forecast_steps)
    print("Forecast for the next 24 steps:", prediction)

    # 可视化结果
    plt.plot(data, label="Historical Data")
    plt.plot(range(len(data), len(data) + forecast_steps), prediction, label="Forecast", color='red')
    plt.legend()
    plt.show()

class RandomForecast:
    def __init__(self, historical_data):
        self.historical_data = historical_data

    def forecast(self, steps):
        # 生成随机预测结果
        random_forecast = np.random.randn(steps)
        return random_forecast

# 创建随机预测模型实例
random_forecast_model = RandomForecast(data)
random_prediction = random_forecast_model.forecast(steps=forecast_steps)
print("Random forecast for the next 24 steps:", random_prediction)

# 可视化随机预测结果
plt.plot(data, label="Historical Data")
plt.plot(range(len(data), len(data) + forecast_steps), prediction, label="ARIMA Forecast", color='red')
plt.plot(range(len(data), len(data) + forecast_steps), random_prediction, label="Random Forecast", color='green')
plt.legend()
plt.show()

