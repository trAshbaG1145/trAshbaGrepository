"""
模型评估脚本 - 评估训练好的模型在验证集/测试集上的性能

【作用】
- 评估训练好的模型在 VisDrone 数据集上的性能
- 计算详细的检测指标和类别级别的 AP
- 生成评估曲线和混淆矩阵

【主要功能】
1. 计算核心指标：mAP@0.5、mAP@0.5:0.95、Precision、Recall
2. 输出各类别 AP：VisDrone 10 类目标的详细性能
3. 生成可视化：PR 曲线、混淆矩阵、预测示例
4. 支持 CLI 参数：灵活指定模型、数据集、设备等

【使用场景】
- 评估单个模型的详细性能
- 收集论文实验数据
- 分析各类别的检测效果
- 对比不同模型在各类别上的表现

【用法】
  # 评估最佳模型
  python eval.py --model runs/ablation/3_yolov11n_p2_dilated/weights/best.pt
  
  # 评估其他模型
  python eval.py --model runs/ablation/1_baseline_yolov11n/weights/best.pt --batch 8
  
  # 在测试集上评估
  python eval.py --model <model_path> --split test --device 0

【输出内容】
  mAP@0.5     : 0.5234
  mAP@0.5:0.95: 0.3456
  Precision   : 0.6789
  Recall      : 0.5432
  各类别 AP@0.5: pedestrian, people, bicycle, car, ...

【特点】
- ✅ 支持 CLI 参数（灵活配置）
- ✅ 详细的类别级别指标
- ✅ 可用于论文实验数据收集
- ✅ 自动生成可视化结果
"""
import argparse
import os
import random
import numpy as np
from ultralytics import YOLO  # type: ignore


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv11n-P2 on VisDrone")
    parser.add_argument(
        "--model",
        default="runs/ablation/3_yolov11n_p2_dilated/weights/best.pt",
        help="Path to trained weights",
    )
    parser.add_argument(
        "--data",
        default="VisDrone.yaml",
        help="Dataset YAML path",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Evaluation image size")
    parser.add_argument("--batch", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="IoU threshold")
    parser.add_argument("--device", default="0", help="Device id, e.g., '0' or 'cpu'")
    parser.add_argument("--split", default="val", help="Dataset split to evaluate")
    return parser.parse_args()


def main():
    """评估训练好的模型在 VisDrone 测试集上的性能"""

    args = parse_args()
    set_seed()

    # 检查模型文件
    if not os.path.exists(args.model):
        print(f"❌ 模型文件不存在: {args.model}")
        print("请先运行训练脚本: python start_train.py")
        return

    print("=" * 60)
    print("📊 模型评估 - YOLOv11n-P2 on VisDrone")
    print("=" * 60)

    # 加载模型
    print(f"\n📦 加载模型: {args.model}")
    model = YOLO(args.model)

    # 在验证集上评估
    print("\n🔍 在验证集上评估...")
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,  # 低置信度阈值以计算完整 PR 曲线
        iou=args.iou,
        device=args.device,
        plots=True,
        save_json=True,
    )

    # 输出关键指标
    print("\n" + "=" * 60)
    print("📈 评估结果")
    print("=" * 60)
    print(f"mAP@0.5     : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Precision   : {metrics.box.mp:.4f}")
    print(f"Recall      : {metrics.box.mr:.4f}")

    # 尝试获取尺度分布指标（AP_Small 是项目核心关注）
    ap_small = getattr(metrics.box, "map_small", None)
    ap_medium = getattr(metrics.box, "map_medium", None)
    ap_large = getattr(metrics.box, "map_large", None)
    if ap_small is not None:
        print(f"AP_Small    : {ap_small:.4f}  (面积 < 32x32 像素，核心指标)")
    if ap_medium is not None:
        print(f"AP_Medium   : {ap_medium:.4f}")
    if ap_large is not None:
        print(f"AP_Large    : {ap_large:.4f}")

    # 计算推理速度与 FPS（基于 metrics.speed 的 inference ms）
    if hasattr(metrics, "speed") and "inference" in metrics.speed:
        infer_ms = metrics.speed["inference"]
        fps = 1000.0 / infer_ms if infer_ms > 0 else 0
        print(f"FPS (估算)  : {fps:.2f}  (基于 RTX 4060 推理耗时 {infer_ms:.2f} ms)")

    print("-" * 60)
    print("⚠️  核心指标提醒：")
    print("   - AP_Small：微小目标(<32x32)的检测精度，是本项目最重要指标。")
    print("   - mAP@0.5：基础检测精度。")
    print("   - mAP@0.5:0.95：高精度定位能力。")
    print("   - FPS：需在 RTX 4060 上确认满足实时性要求。")

    # 按类别输出
    print("\n📊 各类别 AP@0.5:")
    print("-" * 60)
    # 使用模型自带的类别映射，避免与数据集 YAML 不一致
    class_names = getattr(model, "names", None) or {}
    # 按类别索引排序输出
    for idx, ap in enumerate(metrics.box.ap50):
        name = class_names.get(idx, f"class_{idx}") if isinstance(class_names, dict) else str(idx)
        print(f"{idx:2d}. {name:20s}: {ap:.4f}")

    print("\n" + "=" * 60)
    print("✅ 评估完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
