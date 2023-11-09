import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def compute_mean_psnr(test_data, noisy_test_data):
    # Compute the mean PSNR (Peak Signal-to-Noise Ratio) for a set of image pairs

    if len(test_data) != len(noisy_test_data):
        raise ValueError("Input lists must have the same length")

    psnr_values = []

    for original, noisy in zip(test_data, noisy_test_data):
        psnr_current_value = psnr(original, noisy)
        psnr_values.append(psnr_current_value)

    mean_psnr = np.mean(psnr_values)
    return mean_psnr


def compute_mean_ssim(test_data, noisy_test_data):
    # Compute the mean SSIM (Structural Similarity Index) for a set of image pairs

    if len(test_data) != len(noisy_test_data):
        raise ValueError("Input lists must have the same length")

    ssim_values = []

    for original, noisy in zip(test_data, noisy_test_data):
        # Calculate an appropriate win_size based on image dimensions
        win_size = min(original.shape[0], original.shape[1], noisy.shape[0], noisy.shape[1])
        win_size = max(7, win_size)  # Ensure it's at least 7

        ssim_score = ssim(original, noisy, channel_axis=2)
        ssim_values.append(ssim_score)

    mean_ssim = np.mean(ssim_values)
    return mean_ssim


def compute_psnr_ssim_metrics(noisy_dir, restored_dir):
    noisy_images = os.listdir(noisy_dir)
    restored_images = os.listdir(restored_dir)

    num_pairs = min(len(noisy_images), len(restored_images))

    ssim_value = 0
    psnr_value = 0

    for i in range(num_pairs):
        noisy_image_path = os.path.join(noisy_dir, noisy_images[i])
        restored_image_path = os.path.join(restored_dir, restored_images[i])

        noisy_image = cv2.imread(noisy_image_path)
        restored_image = cv2.imread(restored_image_path)

        psnr_value += psnr(restored_image, noisy_image)
        ssim_value += ssim(restored_image, noisy_image, channel_axis=2)

    mean_psnr = psnr_value / num_pairs
    mean_ssim = ssim_value / num_pairs

    return mean_psnr, mean_ssim
