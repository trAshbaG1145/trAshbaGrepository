# YOLOv11n-P2 for VisDrone - 微小目标检测项目

> 基于 YOLOv11n 的无人机航拍微小目标检测，通过 P2 高分辨率检测头和空洞卷积增强性能。

## 📂 项目结构

```
CV/Project/
├── yolo11n.yaml            # Baseline 配置
├── yolov11n-p2.yaml        # P2 高分辨率检测头
├── yolov11n-p2-dilated.yaml # P2 + 空洞卷积（拔高版）
├── VisDrone.yaml           # 数据集配置
├── ablation_study.py       # 消融实验脚本（推荐）
├── start_train.py          # 单模型训练（备用，默认 P2+Dilated）
├── eval.py                 # 模型评估
├── demo_inference.py       # SAHI 推理对比
├── technical_details.md    # 技术原理详解
└── README.md               # 本文件
```

---

## 🚀 快速开始

### 1. 环境准备

```powershell
pip install ultralytics sahi opencv-python
```

### 2. 消融实验训练（3 个模型）

#### 方式 A：单独训练（推荐，灵活控制）

```powershell
# 训练 Baseline 模型
python ablation_study.py train 1

# 训练 P2 模型
python ablation_study.py train 2

# 训练 P2+Dilated 模型
python ablation_study.py train 3

# 对比所有已训练模型
python ablation_study.py compare
```

#### 方式 B：一键训练全部

```powershell
python ablation_study.py train all
```

**说明：**
- 首次运行会自动下载 VisDrone 数据集（1~2 GB）和预训练权重（~6 MB）
- 每个模型训练约 2-3 小时（100 epochs，RTX 4060 8GB）
- 输出保存在 `runs/ablation/*/weights/best.pt`

### 3. 模型评估

```powershell
# 评估最佳模型
python eval.py --model runs/ablation/3_yolov11n_p2_dilated/weights/best.pt

# 或评估其他模型
python eval.py --model runs/ablation/1_baseline_yolov11n/weights/best.pt
```

### 4. SAHI 推理对比（可选）

```powershell
# 运行 SAHI 切片推理 vs 原生 YOLO 对比
python demo_inference.py --model runs/ablation/3_yolov11n_p2_dilated/weights/best.pt
```

结果保存在 `demo_result/`，包含：
- `sahi_result.jpg` - 切片推理结果（微小目标检测更好）
- `native_yolo/` - 原生推理结果（速度更快）

> 兼容性提示：SAHI 目前对 YOLOv11 支持有限，脚本会在 SAHI 不可用或加载失败时自动跳过，直接使用原生 YOLO 推理。

---

## 📊 模型对比

| 模型 | 检测头 | P2 层 | 膨胀卷积 | 参数量 | 预期 mAP@0.5 |
|------|--------|-------|---------|--------|-------------|
| **Baseline** | P3/P4/P5 | ❌ | ❌ | 6.3M | ~40-45% |
| **P2** | P2/P3/P4/P5 | ✅ | ❌ | 6.4M | ~48-52% (+5-7%) |
| **P2+Dilated** | P2/P3/P4/P5 | ✅ | ✅ | 6.5M | ~50-54% (+2-3%) |

**消融实验设计：**
```
Baseline → 衡量基线性能
   ↓
P2 → 衡量 P2 高分辨率头的贡献
   ↓
P2+Dilated → 衡量膨胀卷积的增量贡献
```

---

## 🔬 核心技术改进

### 1. P2 高分辨率检测头

**原理**：在标准 P3/P4/P5 检测头基础上，增加 P2 层（1/4 下采样）

```
传统 YOLO:  P3(1/8) → P4(1/16) → P5(1/32)
改进方案:   P2(1/4) → P3(1/8) → P4(1/16) → P5(1/32)
```

**优势**：
- 保留 4 倍下采样的浅层特征（纹理、边缘）
- 检测 16×16 以上的微小目标
- 对 < 32×32 像素目标召回率提升 5-7%

### 2. 空洞深度卷积（拔高项）

**原理**：通过膨胀卷积核扩大感受野而不增加参数

```
标准卷积 3×3:        空洞卷积 3×3 (dilation=2):
[■][■][■]           [■][ ][■][ ][■]
[■][■][■]           [ ][ ][ ][ ][ ]
[■][■][■]           [■][ ][■][ ][■]
感受野: 3×3          感受野: 5×5
```

**优势**：
- 捕获更多上下文信息（周围道路、建筑）
- 参数量增加 < 0.1M
- 进一步提升微小目标识别率 2-3%

### 3. SAHI 切片推理

**原理**：将高分辨率图像切片后分别推理，再合并结果

```python
get_sliced_prediction(
    image,
    model,
    slice_height=640,
    slice_width=640,
    overlap_ratio=0.2  # 20% 重叠避免边界漏检
)
```

**优势**：
- 解决图像缩放导致的信息损失
- 局部视野中微小目标更清晰
- 适合 > 1920×1080 的高分辨率图像

**详细技术原理**：见 `technical_details.md`

---

## 📁 输出文件说明

训练和推理会生成以下文件：

```
runs/
└── ablation/
    ├── 1_baseline_yolov11n/
    │   ├── weights/best.pt          # 最佳模型
    │   ├── results.csv              # 训练指标
    │   └── results.png              # 训练曲线
    ├── 2_yolov11n_p2/
    │   └── ...
    ├── 3_yolov11n_p2_dilated/
    │   └── ...
    └── results_summary.json         # 对比汇总

demo_result/
├── sahi_result.jpg                  # SAHI 推理结果
└── native_yolo/predict/             # 原生推理结果
```

---

## 🛠️ 常见问题

**Q0: 如何确保实验可复现？**  
脚本默认设置了随机种子（42），如需更改可在入口脚本中调整。注意 AMP 与多线程数据加载仍可能带来微小波动。

**Q1: 训练失败，提示找不到配置文件？**
```powershell
# 确保在项目目录下运行
cd E:\trAshbaGrepository\CV\Project
python ablation_study.py train 1
```

**Q2: 显存不足 (OOM)？**
```python
# 修改 ablation_study.py 第 87 行
batch=8  # 从 16 改为 8
```

**Q3: 数据集下载很慢？**
- 可从 [VisDrone 官网](http://aiskyeye.com/) 手动下载
- 解压到 `datasets/VisDrone/`

**Q3.1: 标注转换会出界吗？**
转换脚本已将框裁剪到图像范围，并过滤严重遮挡/截断样本，减少无效标注对训练的干扰。

**Q4: 想跳过某个模型的训练？**
- 完全可以，只训练需要的模型即可
- 对比时会自动识别已完成的模型

**Q5: 如何修改训练轮数？**
```powershell
# 训练 50 轮（快速实验）
python ablation_study.py train 1 50

# 训练 150 轮（完整实验）
python ablation_study.py train 1 150
```

---

## 📚 参考资源

- **YOLOv11 官方文档**: https://docs.ultralytics.com/models/yolo11/
- **SAHI 项目**: https://github.com/obss/sahi
- **VisDrone 数据集**: http://aiskyeye.com/
- **技术细节文档**: `technical_details.md`

---

## 📝 引用

如果本项目对你有帮助，欢迎引用：

```bibtex
@misc{yolov11n-p2-visdrone,
  title={YOLOv11n-P2: High-Resolution Feature Pyramid for Tiny Object Detection},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/your-repo}}
}
```

---

## ✅ 项目完成标准

- [ ] 完成 3 个模型训练（Baseline, P2, P2+Dilated）
- [ ] 生成对比结果 (`results_summary.json`)
- [ ] 运行 SAHI 推理对比
- [ ] 整理训练曲线和推理结果截图
- [ ] 撰写实验报告（包含 mAP 对比表、可视化结果）

**现在就开始训练吧！** 🚀

