import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

# 1. 生成模拟数据 (假设 1 是“好瓜”，0 是“坏瓜”)
# 这里的特征可以是色泽、根蒂等数值化后的结果
X = np.random.rand(100, 2) 
y = (X[:, 0] + X[:, 1] > 1).astype(int)
noise = np.random.choice([0, 1], size=len(y), p=[0.9, 0.1]) # 10% 的概率翻转标签
y = np.abs(y - noise)

# 2. 留出法划分数据集 (70% 训练, 30% 测试)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 训练模型 (以逻辑回归为例)
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. 在测试集上进行预测
y_pred = model.predict(X_test)

# 5. 计算并打印各项指标
print("--- 性能评估结果 ---")
print(f"精度 (Accuracy):  {accuracy_score(y_test, y_pred):.2f}")
print(f"查准率 (Precision): {precision_score(y_test, y_pred):.2f}")
print(f"查全率 (Recall):    {recall_score(y_test, y_pred):.2f}")
print(f"F1 分数 (F1-Score): {f1_score(y_test, y_pred):.2f}")

# 6. 打印混淆矩阵
print("\n--- 混淆矩阵 (Confusion Matrix) ---")
# 输出格式为: 
# [ [TN, FP],
#   [FN, TP] ]
print(confusion_matrix(y_test, y_pred))

# 7. 一键生成分类报告
print("\n--- 详细分类报告 ---")
print(classification_report(y_test, y_pred))