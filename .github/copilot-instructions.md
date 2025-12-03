# Copilot / agent 指南 — trAshbaRepository

说明：该仓库是一个混合语言的学习资料集合（以练习实现算法与小型深度学习代码为主）。本文件面向自动化代码助手，提供能立即上手的“可执行知识”：项目结构、关键约定、运行/调试示例和注意事项。

主要架构（大局观）
- **多模块混合仓库**：包含 C/C++ 算法实现（`Algorithm/`、`领巾猫Algorithms/` 等）、Python 机器学习/计算机视觉工具包（`CV/dlcv/`）、笔记/实验（`.ipynb`）和若干课程资料。
- **`CV/dlcv` 是可导入的 Python 包**：目录下含 `__init__.py`、`layers.py`、`solver.py`、`classifiers/` 等，代码风格和 API 与 cs231n 教程相似（模块化层、Solver、简单分类器）。修改此包时考虑到包命名空间为 `CV.dlcv`。

关键文件与可复用示例
- `CV/dlcv/layers.py`：基础算子实现（affine、relu、conv、pool、batchnorm 等）。用作数值实现与单元练习的参考。
- `CV/dlcv/solver.py`：训练循环与优化器接口（Solver 接受 model、data 字典并调用 `model.loss`）。文档字符串包含示例用法，AI agent 在修改训练/模型时应参考该 API。
- `CV/dlcv/classifiers/fc_net.py`：示例模型（TwoLayerNet），展示了模型与 `Solver` 的配合方式。
- `CV/dlcv/datasets/get_datasets.sh` 与 `CV/dlcv/datasets/cifar-10-batches-py/`：数据下载/存储位置。脚本为 shell 脚本，Windows 环境需要 WSL/Git Bash 或手动放置数据文件。
- C 源码样例：`Algorithm/` 下大量单文件实现（`binarysearch.c`、`back_track.c` 等），适合快速编译运行验证算法。

项目特定的开发/运行工作流
- Python 快速准备（最小依赖）：仓库中没有 `requirements.txt`。为运行 `CV/dlcv` 中代码，通常至少需要 `numpy` 和 `future`：
  - 创建虚拟环境并安装：``pip install numpy future``
- 运行示例（从仓库根目录）：
  - 导入包并运行最小训练例子（交互式或脚本）：
    ```powershell
    python - <<'PY'
    import numpy as np
    from CV.dlcv.classifiers.fc_net import TwoLayerNet
    from CV.dlcv.solver import Solver

    X_train = np.random.randn(100, 3*32*32)
    y_train = np.random.randint(0, 10, 100)
    X_val = np.random.randn(20, 3*32*32)
    y_val = np.random.randint(0, 10, 20)

    data = {'X_train': X_train, 'y_train': y_train, 'X_val': X_val, 'y_val': y_val}
    model = TwoLayerNet()
    solver = Solver(model, data, num_epochs=1, batch_size=50, print_every=10)
    solver.train()
    PY
    ```
  - 备注：`Solver` 的文档字符串即为最佳示例，优先参考 `CV/dlcv/solver.py`。

- C/C++ 运行（按文件单编译）：Windows 下建议使用 WSL 或 MinGW；示例：
  - WSL/Git Bash / MinGW: ``gcc "Algorithm/领巾猫Algorithms/back_track/back_track.c" -O2 -o back_track.exe``
  - 注意路径中含中文或特殊字符时使用引号或路径转义。

代码风格与项目约定（agent 应遵守）
- Python 风格：代码中使用 `from builtins import range` 和 `future`，说明兼顾 Py2/Py3 的兼容写法；但在编辑时以 Python3 为目标即可，同时保留 `future` 相关导入以避免破坏包兼容性。
- 模块 API：模型类保存可学习参数到 `self.params`（字典），并通过 `loss(X, y)` 在训练/测试模式下切换行为 — 这是与 `Solver` 协作的核心约定。
- 算子实现：`layers.py` 中多个函数采用“返回 (out, cache)” 和“按照 TODO 填充实现”的风格；当修改这些函数，保持返回值契约（cache 的结构）否则会破坏 `layer_utils` 或上层模型。

集成点与外部依赖
- 数据：`CV/dlcv/datasets` 下的 CIFAR-10 文件夹是代码默认数据路径，许多模型假定输入为 `(N, 3*32*32)`。
- 外部库：当前代码明显依赖 `numpy`（`import numpy as np`），并使用 `future` 做兼容性处理。代码库中未见 `torch`/`tensorflow` 的 import（因此不要默认引入深度框架）。

常见编辑场景与示例指令给 Agent
- 在实现/重构 `layers.py` 的函数时：
  - 先读取对应函数的文档注释（header），遵守输入输出/shape 约定；
  - 保持 `cache` 中包含上游 backward 所需的变量（见现有实现样例）；
  - 在修改后通过运行上面的最小训练脚本快速 smoke-test（确保 `solver.train()` 不抛异常）。
- 在新增模型（`classifiers/`）时：
  - 遵循 `TwoLayerNet` 的 `self.params` 约定与 `loss(X,y)` 返回值契约；
  - 若新增超参或训练流程，优先通过 `Solver` 的可配置项（`optim_config`, `lr_decay`, `batch_size` 等）暴露。

注意事项与已知坑
- 仓库文件使用混合编码与中文路径（例如 `领巾猫Algorithms`），在脚本/终端操作时请确保使用 UTF-8 路径处理或用引号包裹路径。
- 没有统一的 requirements / CI，修改依赖或新增可复现脚本时建议同时在仓库根或 `CV/dlcv` 下添加 `requirements.txt`。

如果需我把这些内容进一步细化（例如添加完整的 `requirements.txt`、把常用运行脚本写入 `scripts/`、或补充更多 Notebook 示例），告诉我你优先想要的方向。

---
最后一次检索：仓库根有 `README.md`（简短说明），未发现现成的 agent 指南文件（AGENT.md / copilot-instructions）。
