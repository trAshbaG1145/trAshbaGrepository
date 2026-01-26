# 使用线性规划求解生猪饲料配方问题

import pandas as pd
import numpy as np
from scipy.optimize import linprog
import re
import os

def main():
    # 1. 读取数据
    try:
        df_ing = pd.read_csv('原料营养价值.csv')
        df_std = pd.read_csv('生猪饲养标准.csv')
    except FileNotFoundError:
        print("错误：未找到CSV文件，请确认文件名和路径是否正确。")
        return

    # 2. 数据预处理：原料表
    df_ing.set_index('原料名称', inplace=True)
    df_ing_T = df_ing.T

    # 清洗列名函数
    def clean_name(name):
        name = re.sub(r'[（\(].*?[）\)]', '', name) 
        name = name.replace('，%', '')
        name = name.strip()
        return name

    df_ing_T.columns = [clean_name(col) for col in df_ing_T.columns]
    df_ing_T = df_ing_T.apply(pd.to_numeric, errors='coerce')

    # 提取关键向量
    price = df_ing_T['价格']
    lower_bounds = df_ing_T['用量下限']
    upper_bounds = df_ing_T['用量上限']
    fixed_usage = df_ing_T['等量使用']

    ingredients = df_ing_T.index.tolist()
    n_ingredients = len(ingredients)

    # 3. 数据预处理：标准表
    df_std.set_index('营养指标\\体重阶段', inplace=True)
    df_std.index = [clean_name(idx) for idx in df_std.index]
    
    # 修正名称不一致
    new_index = [name if name != '粗蛋白质' else '粗蛋白' for name in df_std.index]
    df_std.index = new_index

    # 确定需要满足的营养指标
    nutrients = [n for n in df_std.index if n in df_ing_T.columns]
    print(f"参与计算的营养指标: {nutrients}\n")

    # 4. 定义求解函数
    def solve_feed_mix(stage_name):
        requirements = df_std[stage_name]
        c = price.values
        
        # 不等式约束
        A_ub = []
        b_ub = []
        
        for nutrient in nutrients:
            req_val = requirements[nutrient]
            nut_content = df_ing_T[nutrient].fillna(0).values
            A_ub.append(-nut_content)
            b_ub.append(-req_val * 100)
            
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # 等式约束
        A_eq = []
        b_eq = []
        A_eq.append(np.ones(n_ingredients))
        b_eq.append(100.0)
        
        # 变量边界
        bounds = []
        for i in range(n_ingredients):
            lb = lower_bounds.iloc[i]
            ub = upper_bounds.iloc[i]
            fix = fixed_usage.iloc[i]
            
            if fix > 0:
                lb = fix
                ub = fix
                
            bounds.append((lb, ub))
            
        # 求解
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        return res

    # 5. 循环求解并输出到文件
    output_filename = '配方计算结果.txt'
    
    # 使用 'w' 模式打开文件，encoding='utf-8' 防止中文乱码
    with open(output_filename, 'w', encoding='utf-8') as f:
        
        # 定义一个内部函数，同时打印到控制台和写入文件
        def log(text):
            print(text)
            f.write(text + '\n')
            
        stages = ['20-50kg', '50-80kg', '80-120kg']

        for stage in stages:
            log(f"正在计算阶段：{stage} ...")
            res = solve_feed_mix(stage)
            
            log(f"\n===== 结果展示: {stage} =====")
            if res.success:
                total_cost = res.fun / 100
                log(f"求解成功！最低饲料单价: {total_cost:.4f} 元/kg")
                
                mix_series = pd.Series(res.x, index=ingredients)
                valid_mix = mix_series[mix_series > 1e-4].round(4)
                
                log("最佳配方 (%):")
                log(valid_mix.to_string())
                log("-" * 30)
            else:
                log(f"求解失败: {res.message}")
            log("\n")
            
    print(f"所有结果已成功保存到当前目录下的文件：{output_filename}")

if __name__ == "__main__":
    main()