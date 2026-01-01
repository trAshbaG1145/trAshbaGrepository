import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def convert_box(size, box):
    # 将 VisDrone 的 xywh 转换为 YOLO 的 xywh (归一化)
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2] / 2.0) * dw
    y = (box[1] + box[3] / 2.0) * dh
    w = box[2] * dw
    h = box[3] * dh
    return (x, y, w, h)

def visdrone2yolo(dir_path):
    """将 VisDrone 标注转换为 YOLO 格式"""
    img_dir = dir_path / 'images'
    label_dir = dir_path / 'labels'
    anno_dir = dir_path / 'annotations'

    if not anno_dir.exists():
        print(f"⚠️ 跳过: 找不到 annotations 文件夹 -> {anno_dir}")
        return

    # 创建 labels 文件夹
    label_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有标注文件
    anno_files = list(anno_dir.glob('*.txt'))
    print(f"📂 正在转换 {dir_path.name}... 共 {len(anno_files)} 个文件")

    for f in tqdm(anno_files):
        out_file = label_dir / f.name
        
        # 对应的图片路径 (用于获取图片尺寸)
        img_file = img_dir / f.with_suffix('.jpg').name
        if not img_file.exists():
            # 尝试 png 格式
            img_file = img_dir / f.with_suffix('.png').name
        
        if not img_file.exists():
            continue # 如果找不到对应图片，跳过

        try:
            with Image.open(img_file) as img:
                width, height = img.size
        except:
            continue

        with open(f, 'r') as file:
            lines = file.readlines()
            
        with open(out_file, 'w') as file:
            for line in lines:
                data = line.strip().split(',')
                if len(data) < 8:
                    continue
                
                # VisDrone 格式: 
                # <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
                
                category = int(data[5])
                # 过滤掉 "Ignored regions"(0) 和 "Others"(11)
                if category == 0 or category == 11:
                    continue

                truncation = float(data[6])
                occlusion = int(data[7])
                # 可选过滤：极端遮挡或截断的样本干扰训练
                if truncation > 0.7 or occlusion >= 2:
                    continue
                
                # 映射类别 ID (VisDrone 1-10 -> YOLO 0-9)
                # 1:pedestrian -> 0, 2:people -> 1, ..., 10:motor -> 9
                class_id = category - 1 
                
                # 提取并裁剪坐标到图像范围，防止越界或负值
                left = max(0.0, float(data[0]))
                top = max(0.0, float(data[1]))
                right = min(width, left + float(data[2]))
                bottom = min(height, top + float(data[3]))
                w = max(0.0, right - left)
                h = max(0.0, bottom - top)
                if w <= 0 or h <= 0:
                    continue
                box = (left, top, w, h)

                # 转换坐标
                bb = convert_box((width, height), box)
                
                # 写入 YOLO 格式: class x_center y_center w h
                file.write(f"{class_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")

# === 主程序 ===
if __name__ == '__main__':
    # 定义数据集根目录 (根据您的截图结构)
    base_path = Path('datasets/VisDrone')

    # 需要转换的三个数据集目录
    dirs_to_convert = [
        base_path / 'VisDrone2019-DET-train',
        base_path / 'VisDrone2019-DET-val',
        base_path / 'VisDrone2019-DET-test-dev'
    ]

    print("🚀 开始将 VisDrone 格式转换为 YOLO 格式...")
    for d in dirs_to_convert:
        if d.exists():
            visdrone2yolo(d)
        else:
            print(f"⚠️ 目录不存在: {d}")

    print("\n✅ 转换完成！")