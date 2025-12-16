# 实验报告（Exp1）

**日期**: 2025-12-16

---

## 一、实验目的
- 通过Numpy手写实现全连接神经网络（MLP），理解前向与反向传播的矩阵运算本质。
- 掌握参数初始化、激活函数、损失函数、准确率计算、梯度更新（学习率）等关键环节。
- 在`make_moons`二分类数据集上完成训练与测试，并对网络结构、初始化、数据分布与学习率进行实验分析与说明。

---

## 二、实验内容及步骤（代码补全及解释）
> 说明：本节严格依据`experiment_1.ipynb`内容整理，包含网络结构、参数初始化、激活函数、前向/反向传播、损失与准确率、参数更新、训练整合、数据集与实验设置。

### 2.1 网络结构（第3节）
- 以列表`NN_ARCHITECTURE`描述每层：`input_dim`、`output_dim`、`activation`。
```python
NN_ARCHITECTURE = [
  {"input_dim": 2, "output_dim": 25, "activation": "relu"},
  {"input_dim": 25, "output_dim": 50, "activation": "relu"},
  {"input_dim": 50, "output_dim": 50, "activation": "relu"},
  {"input_dim": 50, "output_dim": 25, "activation": "relu"},
  {"input_dim": 25, "output_dim": 1, "activation": "sigmoid"},
]
```
- 要点：相邻层的`input_dim`需与前层`output_dim`一致；可尝试增减层数或维度、替换激活函数比较效果。

### 2.2 参数初始化（第4节）
- 代码位置：`init_layers(nn_architecture, seed=99)`。
- 补全要点：权重矩阵`W^(l)`形状`(n^[l], n^[l-1])`，偏置`b^(l)`形状`(n^[l], 1)`，用小随机数初始化打破对称、避免梯度问题。
```python
params_values['W' + str(layer_idx)] = np.random.randn(layer_output_size, layer_input_size) * 0.1
params_values['b' + str(layer_idx)] = np.random.randn(layer_output_size, 1) * 0.1
```
- 扩展练习：将参数全设为0，观察训练失败原因（对称性无法打破、梯度相同）。

### 2.3 激活函数（第5节）
- 已实现`sigmoid`/`sigmoid_backward`；需补全`relu`/`relu_backward`：
```python
def relu(Z):
  return np.maximum(0, Z)
def relu_backward(dA, Z):
  dZ = np.array(dA, copy=True)
  dZ[Z <= 0] = 0
  return dZ
```
- 要点：ReLU在Z<=0处导数为0；Sigmoid导数在饱和区间趋近0，易梯度消失。

### 2.4 前向传播（第6节）
- 单层：`single_layer_forward_propagation(A_prev, W_curr, b_curr, activation)`
```python
Z_curr = np.dot(W_curr, A_prev) + b_curr
A_curr = relu(Z_curr) 或 sigmoid(Z_curr)
return A_curr, Z_curr
```
- 全网络：`full_forward_propagation(X, params_values, nn_architecture)`，逐层调用单层前向，并缓存`memory["A{idx}"]`与`memory["Z{layer_idx}"]`供反传使用。

### 2.5 损失与准确率（第7节）
- 二分类交叉熵：
```python
cost = -1/m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
```
- 精度计算：概率>0.5判为1，否则0；与标签比较求均值。

### 2.6 反向传播（第8节）
- 单层反传：先`dZ = backward_activation(dA, Z)`，再：
```python
dW = np.dot(dZ, A_prev.T)/m
db = np.sum(dZ, axis=1, keepdims=True)/m
dA_prev = np.dot(W_curr.T, dZ)
```
- 全网络反传：自最后一层起迭代，依据`memory`和`params_values`计算各层`dW、db`并保存到`grads_values`。

### 2.7 参数更新（第9节）
- 按学习率α梯度下降：
```python
W = W - α * dW
b = b - α * db
```

### 2.8 训练整合（第10节）
- `train(X, Y, nn_architecture, epochs, learning_rate)`流程：
  - 初始化参数；循环`epochs`：前向→计算`cost/accuracy`→反向→更新；可每50次打印一次指标。

### 2.9 数据集与实验设置（第11-12节）
- 生成数据：`make_moons(n_samples=1000, noise=0.2, random_state=100)`；按`TEST_SIZE=0.1`划分训练/测试；绘制数据分布图。
- 训练：
```python
params_values = train(X=np.transpose(X_train), Y=np.transpose(y_train.reshape((y_train.shape[0], 1))),
            nn_architecture=NN_ARCHITECTURE, epochs=10000, learning_rate=0.01)
```
- 测试：单次前向传播评估测试集精度；并按提示将`learning_rate`改为`0.1`与`0.001`观察`cost/accuracy`趋势。

---

## 三、问题解答及分析（严格对应Notebook末尾提示）
1) 完成实验并解释各部分代码与结果（含空缺补全）
- 关键补全点：`init_layers`中的`W/b`形状与随机初始化；`relu/relu_backward`的实现；单层与整网前向、整网反向中的矩阵维度与缓存；`update`的梯度下降更新；`train`的训练闭环与指标记录。
- 结果展示：
  - 训练过程`cost`与`accuracy`随迭代的变化（建议每50次打印或绘图）。
  - 测试集精度打印输出：`Test set accuracy: ...`。

2) 按第6.1节图示，画出第2层与第3层在前向/反向时的矩阵运算，并标注维度
- 前向：`Z^(l) = W^(l) A^(l-1) + b^(l)`；`A^(l) = g(Z^(l))`。
  - 维度：`W^(l) ∈ R^{n^[l]×n^[l-1]}`，`A^(l-1) ∈ R^{n^[l-1]×m}`，`b^(l) ∈ R^{n^[l]×1}`，`Z^(l) ∈ R^{n^[l]×m}`。
- 反向：`dZ^(l) = g'(Z^(l)) ⊙ dA^(l)`；`dW^(l) = (1/m) dZ^(l) (A^(l-1))^T`；`db^(l) = (1/m) Σ dZ^(l)`；`dA^(l-1) = (W^(l))^T dZ^(l)`。

3) 不同网络结构的对比（第3节）
- 观察：增深或增宽可提升表达能力，但过深/过宽可能训练更慢或过拟合；更换激活函数（如全Sigmoid）易梯度消失、收敛慢。
- 建议：在固定`epochs`下适度调整结构并记录`cost/accuracy`对比。

5) 初始化为0的实验（第4.3节）
- 现象：训练失败或极慢，梯度一致、对称性无法打破，所有神经元学习到相同表示。
- 结论：需用小随机数初始化以打破对称且避免饱和。

6) 数据分布实验（第11节）
- 噪声`noise`：由`0→0.2→0.4→0.8→1.0`噪声增大，类间可分性下降，训练更难、精度降低；请保存数据分布图与对应训练结果。
- 测试比例`TEST_SIZE`：当`noise=0.8`且`TEST_SIZE=0.98`时，训练集极少、测试集极多，训练不足导致训练/测试精度均下降；与默认(`noise=0.2`,`TEST_SIZE=0.1`)对比分析变化原因。

7) 学习率对比（第12节）
- 在`epochs=10000`下，将`learning_rate`改为`0.1`与`0.001`：
  - `lr=0.1`：收敛更快、在该任务上通常`cost`下降更明显、`accuracy`更高。
  - `lr=0.001`：步长过小，需更多迭代才能达到同等效果，短期`accuracy`偏低。
- 结论：学习率需与任务、迭代轮数匹配；较小lr要配合增大epochs。

---

## 四、实验体会
- 手写Numpy版MLP让前/反向传播与矩阵维度关系更清晰，理解了缓存`A/Z`对反传的必要性。
- 随机小初始化与合适激活函数至关重要；Sigmoid饱和与全零初始化都会显著影响训练。
- 学习率、网络结构、数据噪声与训练/测试划分对最终效果影响显著，需要系统对比与记录。

---

## 需要你操作的部分
- 将以下截图/结果插入本Markdown或Notebook：
  - 数据集分布图（不同`noise`与`TEST_SIZE`）及对应训练结果对比。
  - 训练过程中`cost/accuracy`随迭代的日志或曲线。
  - 学习率`0.1`与`0.001`两组训练对比结论。
- 按第6.1节要求绘制第2/第3层的矩阵维度示意图，并标注维度。
- 可复现性：建议固定随机种子（`np.random.seed(…); random_state=…`）保持数据与初始化一致。
