import os
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

test_results_dir = "C:/Shreyas/AI courseowkr/results1"
ground_truth_dir = "C:/Shreyas/AI courseowkr/png_dataset/test/ct"

if not os.path.exists(test_results_dir) or not os.path.exists(ground_truth_dir):
    raise FileNotFoundError("One or both directories do not exist.")

test_images = sorted(os.listdir(test_results_dir))
ground_truth_images = sorted(os.listdir(ground_truth_dir))

if len(test_images) != len(ground_truth_images):
    raise ValueError(f"Mismatch in image count: {len(test_images)} test images vs {len(ground_truth_images)} ground truth images.")

mae_list, psnr_list, ssim_list, ncc_list = [], [], [], []

def normalized_cross_correlation(img1, img2):
    img1_mean = np.mean(img1)
    img2_mean = np.mean(img2)
    numerator = np.sum((img1 - img1_mean) * (img2 - img2_mean))
    denominator = np.sqrt(np.sum((img1 - img1_mean) ** 2) * np.sum((img2 - img2_mean) ** 2))
    return numerator / denominator

for test_img_name, gt_img_name in zip(test_images, ground_truth_images):
    test_img_path = os.path.join(test_results_dir, test_img_name)
    gt_img_path = os.path.join(ground_truth_dir, gt_img_name)

    test_img = Image.open(test_img_path).convert("L").resize((256, 256))  
    gt_img = Image.open(gt_img_path).convert("L").resize((256, 256))  

    test_array = np.array(test_img, dtype=np.float32)
    gt_array = np.array(gt_img, dtype=np.float32)

    test_array /= 255.0
    gt_array /= 255.0

    mae = np.mean(np.abs(test_array - gt_array))
    psnr_value = psnr(gt_array, test_array, data_range=1)
    ssim_value = ssim(gt_array, test_array, data_range=1)
    ncc_value = normalized_cross_correlation(gt_array, test_array)

    mae_list.append(mae)
    psnr_list.append(psnr_value)
    ssim_list.append(ssim_value)
    ncc_list.append(ncc_value)

print("\nEvaluation Metrics:")
print(f"Mean Absolute Error (MAE): {np.mean(mae_list):.4f}")
print(f"Peak Signal-to-Noise Ratio (PSNR): {np.mean(psnr_list):.4f}")
print(f"Structural Similarity Index Measure (SSIM): {np.mean(ssim_list):.4f}")
print(f"Normalized Cross-Correlation (NCC): {np.mean(ncc_list):.4f}")