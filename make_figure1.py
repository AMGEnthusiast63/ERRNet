import cv2
import numpy as np
import os

# 选择 PSNR 提升最大的样本
sample = "2007_003506"
base_dir = "results/CEILNet_table2"

# 读取四张图
input_img = cv2.imread(f"{base_dir}/{sample}/m_input.png")
baseline_img = cv2.imread(f"{base_dir}/{sample}/baseline.png")
ours_img = cv2.imread(f"{base_dir}/{sample}/attdrnet65.png")
gt_img = cv2.imread(f"{base_dir}/{sample}/t_label.png")

if baseline_img is None:
    print("baseline.png 不存在，请先运行 Baseline 测试")
    exit()

if ours_img is None:
    print("attdrnet65.png 不存在，请检查文件名")
    exit()

# 统一高度
h = min(input_img.shape[0], baseline_img.shape[0], ours_img.shape[0], gt_img.shape[0])
input_img = cv2.resize(input_img, (int(input_img.shape[1] * h / input_img.shape[0]), h))
baseline_img = cv2.resize(baseline_img, (int(baseline_img.shape[1] * h / baseline_img.shape[0]), h))
ours_img = cv2.resize(ours_img, (int(ours_img.shape[1] * h / ours_img.shape[0]), h))
gt_img = cv2.resize(gt_img, (int(gt_img.shape[1] * h / gt_img.shape[0]), h))

# 水平拼接
row = np.hstack([input_img, baseline_img, ours_img, gt_img])

# 添加标签（在图片上方，不遮挡内容）
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.0
thickness = 2

labels = ["Input", "Baseline", "Ours", "GT"]
widths = [input_img.shape[1], baseline_img.shape[1], ours_img.shape[1], gt_img.shape[1]]

# 创建一个带白色背景的顶部区域
top_height = 50
top_bar = np.ones((top_height, row.shape[1], 3), dtype=np.uint8) * 255

# 在顶部区域写文字
x_offset = 10
for i, (label, w) in enumerate(zip(labels, widths)):
    cv2.putText(top_bar, label, (x_offset + (w // 2) - 30, top_height - 15), 
                font, font_scale, (0, 0, 0), thickness)
    x_offset += w

# 垂直拼接顶部区域和图片
result = np.vstack([top_bar, row])

# 保存
os.makedirs("figures", exist_ok=True)
cv2.imwrite("figures/figure1.png", result)
print("Figure 1 saved to figures/figure1.png")