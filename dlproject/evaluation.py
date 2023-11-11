import os
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import numpy as np


def compute_mean_psnr(test_data, reconstructed_test_data):
    if len(test_data) != len(reconstructed_test_data):
        raise ValueError('Input lists must have the same length')

    psnr_values = []
    for original, reconstructed in zip(test_data, reconstructed_test_data):
        psnr_values.append(psnr(original, reconstructed, data_range=255.0))
    return np.mean(psnr_values)


def compute_mean_ssim(test_data, reconstructed_test_data):
    if len(test_data) != len(reconstructed_test_data):
        raise ValueError('Input lists must have the same length')

    ssim_values = []
    for original, reconstructed in zip(test_data, reconstructed_test_data):
        ssim_values.append(ssim(original, reconstructed, data_range=255.0, channel_axis=2))
    return np.mean(ssim_values)


def compute_psnr_ssim_metrics_from_directories(original_dir, restored_dir):
    original_images = sort_paths_by_index(os.listdir(original_dir))
    restored_images = sort_paths_by_index(os.listdir(restored_dir))

    num_pairs = min(len(original_images), len(restored_images))

    ssim_values = []
    psnr_values = []

    for i in range(num_pairs):
        original_image_path = os.path.join(original_dir, original_images[i])
        restored_image_path = os.path.join(restored_dir, restored_images[i])

        original_image = cv2.imread(original_image_path)
        restored_image = cv2.imread(restored_image_path)

        psnr_values.append(psnr(restored_image, original_image))
        ssim_values.append(ssim(restored_image, original_image, channel_axis=2))

    mean_psnr = np.mean(psnr_values)
    mean_ssim = np.mean(ssim_values)
    std_psnr = np.std(psnr_values)
    std_ssim = np.std(ssim_values)

    return mean_psnr, mean_ssim, std_psnr, std_ssim


def sort_paths_by_index(paths):
    return sorted(paths, key=lambda x: int(x.split('_')[-1].split('.')[0]))
