import cv2
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def calculate_ncc(img1, img2):
    img1 = img1.astype(np.float32) / 255.0
    img2 = img2.astype(np.float32) / 255.0
    numerator = np.sum((img1 - np.mean(img1)) * (img2 - np.mean(img2)))
    denominator = np.sqrt(np.sum((img1 - np.mean(img1))**2) * np.sum((img2 - np.mean(img2))**2))
    return numerator / (denominator + 1e-8)

def calculate_lmse(img1, img2, patch_size=8):
    img1 = img1.astype(np.float32) / 255.0
    img2 = img2.astype(np.float32) / 255.0
    h, w = img1.shape[:2]
    mse_sum = 0
    count = 0
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch1 = img1[i:i+patch_size, j:j+patch_size]
            patch2 = img2[i:i+patch_size, j:j+patch_size]
            mse = np.mean((patch1 - patch2) ** 2)
            mse_sum += mse
            count += 1
    return mse_sum / count

def compute_metrics(gt_path, output_path):
    gt = cv2.imread(gt_path)
    out = cv2.imread(output_path)
    
    if gt is None or out is None:
        return None
    
    # 将 GT 缩放到输出图片的尺寸
    h_out, w_out = out.shape[:2]
    gt_resized = cv2.resize(gt, (w_out, h_out))
    
    psnr = peak_signal_noise_ratio(gt_resized, out, data_range=255)
    ssim = structural_similarity(gt_resized, out, channel_axis=2, data_range=255)
    ncc = calculate_ncc(gt_resized, out)
    lmse = calculate_lmse(gt_resized, out)
    
    return {'PSNR': psnr, 'SSIM': ssim, 'NCC': ncc, 'LMSE': lmse}

# 配置路径
gt_dir = "./datasets/raw_data/your_5_photos/gt"
results_dir = "./results/custom"

photo_ids = ["01", "02", "03", "04", "05"]
ext = ".jpg"

print("Photo\t\tBaseline\t\t\t\t\t\tAttDRNet")
print("\t\tPSNR\tSSIM\tNCC\tLMSE\t\tPSNR\tSSIM\tNCC\tLMSE")
print("-" * 100)

for pid in photo_ids:
    gt_path = os.path.join(gt_dir, f"{pid}{ext}")
    baseline_path = os.path.join(results_dir, pid, "errnet.png")
    att_path = os.path.join(results_dir, f"{pid}_", "errnet.png")
    
    baseline_metrics = compute_metrics(gt_path, baseline_path)
    att_metrics = compute_metrics(gt_path, att_path)
    
    if baseline_metrics and att_metrics:
        print(f"{pid}\t\t"
              f"{baseline_metrics['PSNR']:.2f}\t{baseline_metrics['SSIM']:.4f}\t{baseline_metrics['NCC']:.4f}\t{baseline_metrics['LMSE']:.6f}\t\t"
              f"{att_metrics['PSNR']:.2f}\t{att_metrics['SSIM']:.4f}\t{att_metrics['NCC']:.4f}\t{att_metrics['LMSE']:.6f}")