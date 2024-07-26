import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

class LoadForecast:
    def __init__(self, historical_data):
        self.historical_data = historical_data
        self.models = {'MW': None, 'MVAR': None}

    def train_arima(self, column, order=(5, 1, 0)):
        # 创建和训练 ARIMA 模型
        self.models[column] = ARIMA(self.historical_data[column], order=order)
        self.models[column] = self.models[column].fit()
        print(f"ARIMA model trained for {column} with order {order}")
    
    def forecast(self, column, steps):
        if self.models[column] is None:
            raise ValueError(f"Model for {column} is not trained. Call train_arima first.")
        forecast_result = self.models[column].forecast(steps=steps)
        return forecast_result

if __name__ == "__main__":
    # 读取历史负荷数据，确保时间戳被正确解析为日期时间格式
    historical_data = pd.read_csv('/Users/luyaozhang/Desktop/fp/code/DTU/DTU_ADN/load_data.csv', index_col=0, parse_dates=True)
    
    # 确定历史数据的截止时间点
    historical_end_time = pd.to_datetime('2015-09-15 23:00:00')
    
    # 筛选历史数据，从开始时间到历史数据的截止时间点
    historical_data = historical_data.loc[historical_data.index <= historical_end_time]
    
    # 预测从历史数据截止时间点的下一个时间点开始，到未来30天的负荷情况（即720个步长）
    forecast_start_time = historical_end_time
    forecast_end_time = forecast_start_time + pd.DateOffset(days=30)  # 预测未来30天
    
    # 创建和训练负荷预测模型，使用历史数据进行训练
    load_forecast = LoadForecast(historical_data)
    
    # 对每一列数据分别训练 ARIMA 模型
    for column in ['MW', 'MVAR']:
        load_forecast.train_arima(column=column, order=(5, 1, 0))
    
    # 生成预测
    forecast_index = pd.date_range(start=forecast_start_time, end=forecast_end_time, freq='H')
    forecast_steps = len(forecast_index)
    
    # 对每一列数据生成预测
    predictions = {}
    for column in ['MW', 'MVAR']:
        predictions[column] = load_forecast.forecast(column=column, steps=forecast_steps)
    
    # 可视化结果
    plt.figure(figsize=(14, 8))
    for column in ['MW', 'MVAR']:
        plt.subplot(2, 1, 1 if column == 'MW' else 2)
        plt.plot(historical_data.index, historical_data[column], label=f"Historical {column}")
        plt.plot(forecast_index, predictions[column], label=f"Forecast {column}", color='red')
        plt.xlabel('Time')
        plt.ylabel(column)
        plt.title(f'{column} Forecasting for Next 30 Days')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
