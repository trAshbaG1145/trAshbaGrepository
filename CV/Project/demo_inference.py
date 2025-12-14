"""
SAHI 推理对比脚本 - 演示 SAHI 切片推理 vs 原生 YOLO 推理的效果对比

【作用】
- 演示 SAHI 切片推理和原生 YOLO 推理的对比效果
- 验证 P2 模型在高分辨率航拍图像上的微小目标检测能力
- 输出可视化结果和检测数量对比

【主要功能】
1. 方法 1：SAHI 切片推理（适合高分辨率图像，微小目标检测更好）
2. 方法 2：原生 YOLO 推理（速度快，作为对比基准）
3. 输出对比：检测框数量、可视化结果、性能对比
4. 支持 CLI 参数：灵活配置切片大小、重叠率、置信度等

【使用场景】
- 对比 SAHI 和原生推理的效果差异
- 展示微小目标检测能力
- 验证 P2 高分辨率检测头的优势
- 为实验报告生成可视化结果

【用法】
  # 使用默认配置
  python demo_inference.py
  
  # 自定义参数
  python demo_inference.py \
      --model runs/ablation/3_yolov11n_p2_dilated/weights/best.pt \
      --image datasets/VisDrone/.../test_image.jpg \
      --slice-height 640 --slice-width 640 \
      --overlap 0.2 --conf 0.25

【输出位置】
  demo_result/
  ├── sahi_result.jpg          # SAHI 切片推理结果（微小目标更好）
  └── native_yolo/predict/     # 原生 YOLO 推理结果（速度更快）

【对比】
  SAHI 切片推理：
  - ✅ 微小目标召回率更高（适合高分辨率图像）
  - ⚠️ 速度较慢（需要切片和合并）
  
  原生 YOLO 推理：
  - ✅ 速度快（直接推理）
  - ⚠️ 可能漏检微小目标（图像缩放导致信息损失）

【特点】
- ✅ 双推理模式对比（一次运行得到两种结果）
- ✅ SAHI 更适合微小目标检测
- ✅ 输出检测数量对比，便于分析
- ✅ 支持自定义切片参数和置信度阈值
"""
import argparse
import os
import sys
from pathlib import Path
from ultralytics import YOLO  # type: ignore
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


def parse_args():
    parser = argparse.ArgumentParser(description="SAHI + YOLO inference demo")
    parser.add_argument(
        "--model",
        default="runs/ablation/3_yolov11n_p2_dilated/weights/best.pt",
        help="Path to trained weights",
    )
    parser.add_argument(
        "--image",
        default="datasets/VisDrone/VisDrone2019-DET-test-dev/images/0000006_00159_d_0000005.jpg",
        help="Image path for inference",
    )
    parser.add_argument("--output", default="demo_result", help="Output directory")
    parser.add_argument("--slice-height", type=int, default=640, help="Slice height for SAHI")
    parser.add_argument("--slice-width", type=int, default=640, help="Slice width for SAHI")
    parser.add_argument("--overlap", type=float, default=0.2, help="Slice overlap ratio")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", default="cuda:0", help="Device id, e.g., 'cuda:0' or 'cpu'")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # ---------------------------------------------------------
    # 检查文件
    # ---------------------------------------------------------
    if not os.path.exists(args.model):
        print(f"❌ 错误: 模型文件不存在: {args.model}")
        print("\n💡 请先运行训练脚本:")
        print("   python start_train.py")
        sys.exit(1)

    image_path = args.image
    if not os.path.exists(image_path):
        print(f"⚠️  警告: 测试图像不存在: {image_path}")
        print("   将尝试使用数据集中的第一张图像...")
        
        # 尝试查找任意测试图像
        test_dir = Path("datasets/VisDrone/VisDrone2019-DET-test-dev/images")
        if test_dir.exists():
            images = list(test_dir.glob("*.jpg"))
            if images:
                image_path = str(images[0])
                print(f"✅ 找到测试图像: {image_path}")
            else:
                print("❌ 错误: 未找到任何测试图像")
                sys.exit(1)
        else:
            print("❌ 错误: 数据集目录不存在")
            print("\n💡 请先运行训练脚本下载数据集:")
            print("   python start_train.py")
            sys.exit(1)
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    print("=" * 60)
    print("🔍 SAHI 切片推理演示")
    print("=" * 60)
    print(f"📦 模型: {args.model}")
    print(f"🖼️  图像: {image_path}")
    print(f"✂️  切片大小: {args.slice_height}x{args.slice_width}")
    print(f"🔗 重叠率: {args.overlap * 100}%")
    print(f"🎯 置信度阈值: {args.conf}")
    print("=" * 60)
    
    # ---------------------------------------------------------
    # 方法 1: 使用 SAHI (推荐用于高分辨率图像)
    # ---------------------------------------------------------
    print("\n🚀 方法 1: SAHI 切片推理 (适用于微小目标)")
    print("-" * 60)
    
    try:
        # 配置 SAHI 模型接口
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",  # SAHI 目前使用 v8 接口 (v11 兼容)
            model_path=args.model,
            confidence_threshold=args.conf,
            device=args.device,
        )
        
        # 执行切片推理
        print("正在执行切片推理...")
        result = get_sliced_prediction(
            image_path,
            detection_model,
            slice_height=args.slice_height,
            slice_width=args.slice_width,
            overlap_height_ratio=args.overlap,
            overlap_width_ratio=args.overlap,
            verbose=1
        )
        
        # 保存结果
        sahi_output = os.path.join(args.output, "sahi_result.jpg")
        result.export_visuals(export_dir=args.output)
        print(f"✅ SAHI 推理完成! 检测到 {len(result.object_prediction_list)} 个目标")
        print(f"📁 结果已保存到: {args.output}/")
        
    except Exception as e:
        print(f"❌ SAHI 推理失败: {e}")
        print("   可能原因: SAHI 版本不兼容或模型格式问题")
    
    # ---------------------------------------------------------
    # 方法 2: 原生 YOLO 推理 (对比基准)
    # ---------------------------------------------------------
    print("\n🚀 方法 2: 原生 YOLO 推理 (无切片)")
    print("-" * 60)
    
    try:
        model = YOLO(args.model)
        
        # 直接推理
        print("正在执行标准推理...")
        results = model.predict(
            image_path,
            conf=args.conf,
            imgsz=640,
            save=True,
            project=args.output,
            name="native_yolo",
            exist_ok=True
        )

        # YOLO 结果中 boxes 可能为空，安全地统计检测数量
        det_count = 0
        if results and results[0].boxes is not None:
            det_count = len(results[0].boxes)

        print(f"✅ 原生推理完成! 检测到 {det_count} 个目标")
        print(f"📁 结果已保存到: {args.output}/native_yolo/")
        
    except Exception as e:
        print(f"❌ 原生推理失败: {e}")
    
    # ---------------------------------------------------------
    # 结果对比
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("📊 推理结果对比")
    print("=" * 60)
    print("💡 建议:")
    print("   - SAHI 方法适用于高分辨率图像 (>1920x1080)")
    print("   - 原生方法更快，但可能漏检微小目标")
    print("   - 对比两种方法的检测框数量和位置")
    print("\n✅ 演示完成! 请查看输出目录:")
    print(f"   {os.path.abspath(args.output)}")
    print("=" * 60)

if __name__ == "__main__":
    main()