from DTU_ADN import DTU_ADN
from datetime import datetime
import os
import numpy as np
import pandas as pd
from pypower.api import ppoption
import matplotlib.pyplot as plt
from scipy.stats import norm

# 加载主网络
net = DTU_ADN()

# 初始化并连接10kV-400V网络
network_number = np.array([28])          # 次级母线
transformer_secondary = network_number   # 次级母线
transformer_primary = np.array([2])      # 初级母线
transformer_branch = np.array([5])       # 变压器支路

# 将 10kV-400 V 网络连接到主网络
for inet in network_number:
    net.connect_10kV400V_network(inet)

# 初始化24小时 
start_time = datetime(year=2015, month=7, day=9, hour=0, minute=0)
end_time = datetime(year=2015, month=7, day=9, hour=23, minute=0)  

# 初始化时间序列索引
net.init_timeseries_index(start_time, end_time)

# 设置电力流选项
opt = ppoption(ENFORCE_Q_LIMS=1, VERBOSE=0, OUT_ALL=0)

# 获取时间索引
t0 = net.start_idx[0]
tn = net.end_idx[0]

# 存储结果的列表
results = []

# 运行电力流计算
# t0到tn之间
for ti in range(t0, tn): 
    # 加载 60kV 网络时间戳 t0 的负载和生成
    net.gen_and_demand_net_60(ti)
    # 加载 60-10-0.4kV 网络在 t0 时的负载和生成时间戳
    net.gen_and_demand_net(ti, network_number)
    out_net_60, success_60 = net.runpf(net.net_60, ppopt=opt)
    out_net, success = net.runpf(net.net, ppopt=opt)
    
    if not success_60 or not success:
        print(f"Power flow failed at time index {ti}")
    else:
        print(f"Power flow succeeded at time index {ti}")


# 读取负荷数据
load_data_file = '/Users/luyaozhang/Desktop/fp/code/DTU/DTU_ADN/load_data.csv'
load_data = pd.read_csv(load_data_file, index_col=0, parse_dates=True)

# 提取 MW 和 MVAR 数据
mw_data = load_data['MW']
mvar_data = load_data['MVAR']

# 计算 MW 数据的统计分析
mean_mw = mw_data.mean()
std_mw = mw_data.std()
mu_mw, sigma_mw = norm.fit(mw_data)

# 计算 MVAR 数据的统计分析
mean_mvar = mvar_data.mean()
std_mvar = mvar_data.std()
mu_mvar, sigma_mvar = norm.fit(mvar_data)

# 绘制 MW 数据的直方图及其正态分布拟合曲线
plt.figure(figsize=(14, 12))

plt.subplot(2, 1, 1)
plt.hist(mw_data, bins=30, density=True, alpha=0.6, color='g', label='MW Load Data')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p_mw = norm.pdf(x, mu_mw, sigma_mw)
plt.plot(x, p_mw, 'k', linewidth=2, label='Normal fit')
plt.xlabel('Power (MW)')
plt.ylabel('Probability Density')
plt.title('MW Load Data and Normal Distribution Fit')
plt.legend()
plt.grid(True)

# 绘制 MVAR 数据的直方图及其正态分布拟合曲线
plt.subplot(2, 1, 2)
plt.hist(mvar_data, bins=30, density=True, alpha=0.6, color='b', label='MVAR Load Data')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p_mvar = norm.pdf(x, mu_mvar, sigma_mvar)
plt.plot(x, p_mvar, 'k', linewidth=2, label='Normal fit')
plt.xlabel('Reactive Power (MVAR)')
plt.ylabel('Probability Density')
plt.title('MVAR Load Data and Normal Distribution Fit')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 输出统计结果
print(f"MW Mean: {mean_mw}")
print(f"MW Standard Deviation: {std_mw}")
print(f"MW Normal Distribution Fit: mu={mu_mw}, sigma={sigma_mw}")

print(f"MVAR Mean: {mean_mvar}")
print(f"MVAR Standard Deviation: {std_mvar}")
print(f"MVAR Normal Distribution Fit: mu={mu_mvar}, sigma={sigma_mvar}")
