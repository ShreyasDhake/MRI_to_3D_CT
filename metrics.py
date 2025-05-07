import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image

generated_dir = "C:/Shreyas/AI courseowkr/pytorch-CycleGAN-and-pix2pix/results/mri_to_ct_pix2pix/test_latest/images/fake_B"  
ground_truth_dir = "C:/Shreyas/AI courseowkr/pytorch-CycleGAN-and-pix2pix/results/mri_to_ct_pix2pix/test_latest/images/real_B"  

mae_list = []
psnr_list = []
ssim_list = []
ncc_list = []

def normalized_cross_correlation(img1, img2):
    img1_mean = np.mean(img1)
    img2_mean = np.mean(img2)
    numerator = np.sum((img1 - img1_mean) * (img2 - img2_mean))
    denominator = np.sqrt(np.sum((img1 - img1_mean) ** 2) * np.sum((img2 - img2_mean) ** 2))
    return numerator / denominator

generated_files = sorted(os.listdir(generated_dir))
ground_truth_files = sorted(os.listdir(ground_truth_dir))

assert len(generated_files) == len(ground_truth_files), "Mismatch in number of images!"

for gen_file, gt_file in zip(generated_files, ground_truth_files):
    gen_path = os.path.join(generated_dir, gen_file)
    gt_path = os.path.join(ground_truth_dir, gt_file)

    gen_img = np.array(Image.open(gen_path).convert("L"), dtype=np.float32)
    gt_img = np.array(Image.open(gt_path).convert("L"), dtype=np.float32)

    mae = np.mean(np.abs(gen_img - gt_img))
    psnr_value = psnr(gt_img, gen_img, data_range=gt_img.max() - gt_img.min())
    ssim_value = ssim(gt_img, gen_img, data_range=gt_img.max() - gt_img.min())
    ncc_value = normalized_cross_correlation(gt_img, gen_img)

    mae_list.append(mae)
    psnr_list.append(psnr_value)
    ssim_list.append(ssim_value)
    ncc_list.append(ncc_value)

mean_mae = np.mean(mae_list)
mean_psnr = np.mean(psnr_list)
mean_ssim = np.mean(ssim_list)
mean_ncc = np.mean(ncc_list)

# Print results
print("Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mean_mae:.4f}")
print(f"Peak Signal-to-Noise Ratio (PSNR): {mean_psnr:.4f}")
print(f"Structural Similarity Index Measure (SSIM): {mean_ssim:.4f}")
print(f"Normalized Cross-Correlation (NCC): {mean_ncc:.4f}")