import cv2
import numpy as np
import os

# 选择的样本列表（提升最明显的几个）
samples = ["89-m", "87-m", "93-m"]
base_dir = "results/wild"

os.makedirs("figures", exist_ok=True)

for idx, sample in enumerate(samples, 1):
    # 读取四张图
    input_img = cv2.imread(f"{base_dir}/{sample}/m_input.png")
    baseline_img = cv2.imread(f"{base_dir}/{sample}/baseline.png")
    ours_img = cv2.imread(f"{base_dir}/{sample}/attdrnet65.png")
    gt_img = cv2.imread(f"{base_dir}/{sample}/t_label.png")

    # 检查是否读取成功
    if input_img is None:
        print(f"跳过 {sample}: m_input.png 不存在")
        continue
    if baseline_img is None:
        print(f"跳过 {sample}: baseline.png 不存在")
        continue
    if ours_img is None:
        print(f"跳过 {sample}: attdrnet65.png 不存在")
        continue
    if gt_img is None:
        print(f"跳过 {sample}: t_label.png 不存在")
        continue

    # 统一高度
    h = min(input_img.shape[0], baseline_img.shape[0], ours_img.shape[0], gt_img.shape[0])
    input_img = cv2.resize(input_img, (int(input_img.shape[1] * h / input_img.shape[0]), h))
    baseline_img = cv2.resize(baseline_img, (int(baseline_img.shape[1] * h / baseline_img.shape[0]), h))
    ours_img = cv2.resize(ours_img, (int(ours_img.shape[1] * h / ours_img.shape[0]), h))
    gt_img = cv2.resize(gt_img, (int(gt_img.shape[1] * h / gt_img.shape[0]), h))

    # 水平拼接
    row = np.hstack([input_img, baseline_img, ours_img, gt_img])

    # 添加标签（在图片上方）
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    labels = ["Input", "Baseline", "Ours", "GT"]
    widths = [input_img.shape[1], baseline_img.shape[1], ours_img.shape[1], gt_img.shape[1]]

    # 创建顶部标签区域
    top_height = 40
    top_bar = np.ones((top_height, row.shape[1], 3), dtype=np.uint8) * 255

    x_offset = 10
    for i, (label, w) in enumerate(zip(labels, widths)):
        text_x = x_offset + (w // 2) - 30
        cv2.putText(top_bar, label, (text_x, top_height - 12), font, font_scale, (0, 0, 0), thickness)
        x_offset += w

    # 垂直拼接
    result = np.vstack([top_bar, row])

    # 保存
    output_path = f"figures/figure2_{sample}.png"
    cv2.imwrite(output_path, result)
    print(f"Saved: {output_path}")

print("所有图片已生成到 figures/ 目录")