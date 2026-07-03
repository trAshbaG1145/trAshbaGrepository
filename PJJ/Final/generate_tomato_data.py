import pandas as pd
import numpy as np
import os

# ==========================================
# 1. 农业合成数据生成脚本
# ==========================================
print("正在生成温室番茄产量数据集...")

# 设置随机种子以保证每次生成的数据相同
np.random.seed(42)
n_samples = 600

# 生成合理的农业环境特征
# 温度: 18-30°C, 湿度: 50-85%, 二氧化碳: 400-1000ppm, 光照: 8-14 小时, 营养液电导率(EC): 1.5-3.0 dS/m
temp_avg = np.random.uniform(18.0, 30.0, n_samples)
humidity_avg = np.random.uniform(50.0, 85.0, n_samples)
co2_ppm = np.random.uniform(400, 1000, n_samples)
light_hours = np.random.uniform(8.0, 14.0, n_samples)
nutrient_ec = np.random.uniform(1.5, 3.0, n_samples)

# 根据非线性生物学关系和随机噪声计算产量 (Target)
# 假设最适宜温度约为 24°C, 最适宜湿度约为 65%
yield_base = 10.0
yield_temp = -0.2 * (temp_avg - 24)**2 + 6
yield_hum = -0.05 * (humidity_avg - 65)**2 + 4
yield_co2 = 0.008 * co2_ppm
yield_light = 0.9 * light_hours
yield_ec = 2.5 * nutrient_ec

# 总产量 (kg/sqm) 加入随机噪声模拟真实世界的方差
yield_kg_sqm = (yield_base + yield_temp + yield_hum + yield_co2 + 
                yield_light + yield_ec + np.random.normal(0, 1.5, n_samples))

# 创建 DataFrame
df = pd.DataFrame({
    'Sample_ID': range(1, n_samples + 1),
    'Temp_avg': temp_avg,
    'Humidity_avg': humidity_avg,
    'CO2_ppm': co2_ppm,
    'Light_Hours': light_hours,
    'Nutrient_EC': nutrient_ec,
    'Yield_kg_sqm': yield_kg_sqm
})

# ---------------------------------------------------------
# 保存 CSV 文件到当前脚本所在的同一文件夹
# ---------------------------------------------------------
# 1. 获取当前脚本所在的绝对目录路径 (确保在IDE运行或命令行运行都能准确定位)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 拼接文件名
file_path = os.path.join(current_dir, 'tomato_yield.csv')

# 3. 导出为 CSV
df.to_csv(file_path, index=False)

print(f"✅ 数据生成成功！")
print(f"📁 文件已保存至: {file_path}")