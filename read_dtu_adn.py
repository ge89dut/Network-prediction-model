from DTU_ADN import DTU_ADN
from datetime import datetime
import os
import numpy as np
import pandas as pd
from pypower.api import ppoption
import matplotlib.pyplot as plt

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

# 添加 get_load_data 方法
def get_load_data():
    # 这里假设负荷数据保存在 net 对象的某个属性中
    # 例如，net.load_data 是一个 pandas Series 或 numpy array
    return net.load_data

# 将 get_load_data 方法添加到 net 对象中
setattr(net, 'get_load_data', get_load_data)


# 定义数据目录路径
data_dir = '/Users/luyaozhang/Desktop/fp/code/DTU/DTU_ADN/timeseries_400V'  # 请更新为实际数据目录路径
output_file = '/Users/luyaozhang/Desktop/fp/code/DTU/DTU_ADN/all_load_data.csv'  # 请更新为实际输出文件路径

# 确保输出目录存在
output_dir = os.path.dirname(output_file)
os.makedirs(output_dir, exist_ok=True)

# 初始化一个空的 DataFrame 用于存储所有负荷数据
load_data_MW = pd.DataFrame()
load_data_MVAR = pd.DataFrame()

# 遍历数据目录中的所有 CSV 文件
for file_name in os.listdir(data_dir):
    if file_name.endswith('.csv'):
        file_path = os.path.join(data_dir, file_name)
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        # 去除列名的前后空白字符
        df.columns = df.columns.str.strip()

        # 提取所有以 '_pu_MW' 结尾的列
        MW_columns = [col for col in df.columns if col.endswith('_pu_MW')]
        if MW_columns:
            df_MW = df[MW_columns]
            # 将数据合并到 load_data 中
            load_data_MW = pd.concat([load_data_MW, df_MW], axis=0)

        # 提取所有以 '_pu_MVAR' 结尾的列
        MVAR_columns = [col for col in df.columns if col.endswith('_pu_MVAR')]
        if MVAR_columns:
            df_MVAR = df[MVAR_columns]
            # 将数据合并到 load_data 中
            load_data_MVAR = pd.concat([load_data_MVAR, df_MVAR], axis=0)

# 合并 MW 和 MVAR 数据
combined_data = pd.concat([load_data_MW, load_data_MVAR], axis=1)

print(df.dtypes)

# 计算合并后数据的每列平均值
average_load_data = combined_data.groupby(combined_data.index).mean()


# 将平均负荷数据保存到一个新的 CSV 文件中
average_load_data.to_csv(output_file)

print(f"Aggregated load data saved to {output_file}")

import pandas as pd

# 定义数据文件路径
all_load_data_file = '/Users/luyaozhang/Desktop/fp/code/DTU/DTU_ADN/all_load_data.csv'
output_file = '/Users/luyaozhang/Desktop/fp/code/DTU/DTU_ADN/load_data.csv'

# 读取现有的所有负荷数据文件
load_data = pd.read_csv(all_load_data_file, index_col=0, parse_dates=True)

# 提取 '_pu_MW' 和 '_pu_MWAR' 列
MW_columns = [col for col in load_data.columns if col.endswith('_pu_MW')]
MWAR_columns = [col for col in load_data.columns if col.endswith('_pu_MVAR')]

# 计算每个时间点的 MW 列的平均值
average_MW = load_data[MW_columns].mean(axis=1)
# 计算每个时间点的 MVAR 列的平均值
average_MVAR = load_data[MVAR_columns].mean(axis=1)

# 创建一个新的 DataFrame 用于存储结果
result_df = pd.DataFrame({
    'Time': load_data.index,
    'MW': average_MW,
    'MVAR': average_MVAR
})

# 将结果保存到新的 CSV 文件中
result_df.to_csv(output_file, index=False)

print(f"Average load data saved to {output_file}")





