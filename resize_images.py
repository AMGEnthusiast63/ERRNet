import cv2
import os

# 源目录和目标目录
src_dir = "datasets/raw_data/your_5_photos"
dst_dir = "datasets/raw_data/your_5_photos_small"

# 目标尺寸（最长边）
target_max_size = 512

# 需要处理的子文件夹
subdirs = ["input", "gt"]

for sub in subdirs:
    src_sub = os.path.join(src_dir, sub)
    dst_sub = os.path.join(dst_dir, sub)
    
    if not os.path.exists(src_sub):
        print(f"跳过 {src_sub}: 不存在")
        continue
    
    os.makedirs(dst_sub, exist_ok=True)
    
    # 遍历所有图片
    for filename in os.listdir(src_sub):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        src_path = os.path.join(src_sub, filename)
        img = cv2.imread(src_path)
        
        if img is None:
            print(f"无法读取: {src_path}")
            continue
        
        # 计算缩放比例
        h, w = img.shape[:2]
        scale = target_max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 缩放
        resized = cv2.resize(img, (new_w, new_h))
        
        # 保存
        dst_path = os.path.join(dst_sub, filename)
        cv2.imwrite(dst_path, resized)
        print(f"已处理: {filename} ({w}x{h} -> {new_w}x{new_h})")

print("完成!")