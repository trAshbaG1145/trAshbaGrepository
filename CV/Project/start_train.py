"""
单模型训练脚本 - 训练单个指定的模型配置（备用）

【作用】
- 训练单个指定的模型配置（默认 yolov11n-p2-dilated.yaml）
- 完整的训练流程：加载预训练权重 → 训练 → 保存
- 针对 8GB 显存优化

【主要功能】
1. 训练单个模型（默认 P2 版本）
2. 自动下载预训练权重和数据集
3. 详细的训练日志和可视化
4. 早停、学习率调整等训练策略

【使用场景】
- 备用：适合快速测试单个配置
- 简单：代码清晰，易于修改训练参数
- 注意：通常情况下推荐使用 ablation_study.py

【用法】
    python start_train.py                          # 训练 P2+Dilated 模型

【输出位置】
  runs/detect/visdrone_yolov11n_p2/weights/best.pt

【对比】
  vs ablation_study.py:
  - ablation_study.py: 支持多模型训练和自动对比 ⭐ 推荐
  - start_train.py: 只能训练单个模型，但代码更简洁
"""
import os
import random
import numpy as np
from ultralytics import YOLO


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def main():
    """
    YOLOv11n-P2 训练脚本 for VisDrone 微小目标检测
    针对 RTX 4060 Laptop 8GB 显存优化
    """
    set_seed()
    
    # ---------------------------------------------------------
    # ⚠️ 配置参数 (DEBUG 模式)
    # ---------------------------------------------------------
    # 注意：正式实验请务必使用 'ablation_study.py' 以确保实验条件一致。
    # 本脚本仅用于快速测试单个模型的连通性或调试报错。
    
    CONFIG_PATH = "yolov11n-p2-dilated.yaml"      # 默认使用含空洞卷积的拔高版
    DATA_PATH = "VisDrone.yaml"           # 数据集配置
    PRETRAIN_WEIGHTS = "yolo11n.pt"       # 预训练权重 (可选)
    
    # 训练超参数
    EPOCHS = 100                           # 训练轮数
    IMGSZ = 640                           # 输入图像大小
    BATCH = 16                            # 批次大小 (显存不足会自动调整)
    DEVICE = 0                            # GPU 设备 ID
    
    # 输出配置
    PROJECT = "runs/detect"
    NAME = "visdrone_yolov11n_p2_dilated"
    
    # ---------------------------------------------------------
    # 检查配置文件
    # ---------------------------------------------------------
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"❌ 模型配置文件不存在: {CONFIG_PATH}")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ 数据集配置文件不存在: {DATA_PATH}")
    
    print("=" * 60)
    print("🚀 YOLOv11n-P2-DILATED 训练任务启动")
    print("=" * 60)
    print(f"📋 模型配置: {CONFIG_PATH}")
    print(f"📊 数据集配置: {DATA_PATH}")
    print(f"🔧 训练参数: Epochs={EPOCHS}, Batch={BATCH}, ImgSize={IMGSZ}")
    print("=" * 60)
    
    # ---------------------------------------------------------
    # 初始化模型
    # ---------------------------------------------------------
    print("\n🔨 正在初始化模型...")
    model = YOLO(CONFIG_PATH)
    
    # 尝试加载预训练权重 (可选，加速收敛)
    if os.path.exists(PRETRAIN_WEIGHTS):
        try:
            model.load(PRETRAIN_WEIGHTS)
            print(f"✅ 成功加载预训练权重: {PRETRAIN_WEIGHTS}")
            print("   (部分层权重因架构改动无法加载是正常现象)")
        except Exception as e:
            print(f"⚠️  预训练权重加载失败: {e}")
            print("   将使用随机初始化权重")
    else:
        print(f"⚠️  未找到预训练权重文件: {PRETRAIN_WEIGHTS}")
        print("   首次运行将自动下载 (约 5MB)")
    
    # ---------------------------------------------------------
    # 开始训练
    # ---------------------------------------------------------
    print("\n🎯 开始训练...")
    results = model.train(
        data=DATA_PATH,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        name=NAME,
        project=PROJECT,
        
        # 优化参数 (针对 8GB 显存)
        workers=4,                    # 数据加载线程
        amp=True,                     # 自动混合精度 (FP16) - 节省显存
        cache=False,                  # 不缓存图像到内存 (节省内存)
        
        # 训练策略
        patience=30,                  # 早停耐心值
        save_period=10,               # 每 10 epochs 保存一次
        
        # 数据增强
        hsv_h=0.015,                  # 色调抖动
        hsv_s=0.7,                    # 饱和度抖动
        hsv_v=0.4,                    # 亮度抖动
        degrees=0.0,                  # 旋转角度 (航拍图不需要大角度旋转)
        translate=0.1,                # 平移
        scale=0.5,                    # 缩放
        mosaic=1.0,                   # Mosaic 数据增强
        
        # 其他
        exist_ok=True,                # 允许覆盖同名实验
        verbose=True,                 # 详细日志
        plots=True                    # 生成训练可视化图表
    )
    
    print("\n" + "=" * 60)
    print("🎉 训练完成!")
    print("=" * 60)
    print(f"📁 最佳模型保存在: {PROJECT}/{NAME}/weights/best.pt")
    print(f"📈 训练日志保存在: {PROJECT}/{NAME}/")
    print("\n💡 下一步:")
    print(f"   1. 查看训练曲线: {PROJECT}/{NAME}/results.png")
    print(f"   2. 验证模型: yolo val model={PROJECT}/{NAME}/weights/best.pt data={DATA_PATH}")
    print(f"   3. 运行 SAHI 推理: python demo_inference.py")
    print("=" * 60)

if __name__ == "__main__":
    main()