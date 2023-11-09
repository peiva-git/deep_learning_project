import os
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import numpy as np


def mean_psnr(test_data, reconstructed_test_data):
    if len(test_data) != len(reconstructed_test_data):
        raise ValueError('Input lists must have the same length')

    psnr_values = []
    for original, reconstructed in zip(test_data, reconstructed_test_data):
        psnr_values.append(psnr(original, reconstructed, data_range=255.0))
    return np.mean(psnr_values)


def mean_ssim(test_data, reconstructed_test_data):
    if len(test_data) != len(reconstructed_test_data):
        raise ValueError('Input lists must have the same length')

    ssim_values = []
    for original, reconstructed in zip(test_data, reconstructed_test_data):
        ssim_values.append(ssim(original, reconstructed, data_range=255.0))
    return np.mean(ssim_values)


def compute_psnr_ssim_metrics_from_directories(noisy_dir, restored_dir):
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

    mean_computed_psnr = psnr_value / num_pairs
    mean_computed_ssim = ssim_value / num_pairs

    return mean_computed_psnr, mean_computed_ssim
