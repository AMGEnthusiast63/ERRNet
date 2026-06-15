import cv2
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def calculate_psnr_ssim(gt_path, output_path):
    gt = cv2.imread(gt_path)
    out = cv2.imread(output_path)
    if gt is None or out is None:
        return None, None
    if gt.shape != out.shape:
        out = cv2.resize(out, (gt.shape[1], gt.shape[0]))
    psnr = peak_signal_noise_ratio(gt, out)
    ssim = structural_similarity(gt, out, channel_axis=2)
    return psnr, ssim

# Wild 数据集路径
wild_dir = "results/wild"
samples = os.listdir(wild_dir)

print("Sample\t\tBaseline PSNR\tOurs PSNR\tDiff\tBaseline SSIM\tOurs SSIM\tDiff")
print("-" * 90)

results = []

for sample in samples:
    sample_path = os.path.join(wild_dir, sample)
    if not os.path.isdir(sample_path):
        continue
    
    gt_path = os.path.join(sample_path, "t_label.png")
    baseline_path = os.path.join(sample_path, "baseline.png")
    ours_path = os.path.join(sample_path, "attdrnet65.png")
    
    if not all(os.path.exists(p) for p in [gt_path, baseline_path, ours_path]):
        print(f"跳过 {sample}: 缺少文件")
        continue
    
    psnr_b, ssim_b = calculate_psnr_ssim(gt_path, baseline_path)
    psnr_o, ssim_o = calculate_psnr_ssim(gt_path, ours_path)
    
    if psnr_b and psnr_o:
        psnr_diff = psnr_o - psnr_b
        ssim_diff = ssim_o - ssim_b
        results.append((sample, psnr_b, psnr_o, psnr_diff, ssim_b, ssim_o, ssim_diff))
        print(f"{sample}\t\t{psnr_b:.2f}\t\t{psnr_o:.2f}\t\t{psnr_diff:+.2f}\t\t{ssim_b:.4f}\t\t{ssim_o:.4f}\t\t{ssim_diff:+.4f}")

# 找出提升最大的样本
if results:
    print("\n" + "=" * 90)
    print("PSNR 提升最大的样本:")
    best_psnr = max(results, key=lambda x: x[3])
    print(f"{best_psnr[0]}: PSNR {best_psnr[1]:.2f} -> {best_psnr[2]:.2f} (+{best_psnr[3]:.2f} dB)")

    print("\nSSIM 提升最大的样本:")
    best_ssim = max(results, key=lambda x: x[6])
    print(f"{best_ssim[0]}: SSIM {best_ssim[4]:.4f} -> {best_ssim[5]:.4f} (+{best_ssim[6]:+.4f})")
else:
    print("没有找到完整的文件，请确保:")
    print("  - 先运行测试: python test_errnet.py --name baseline --dataset wild ...")
    print("  - 以及: python test_errnet.py --name attdrnet65 --dataset wild ...")