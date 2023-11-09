from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim

import numpy as np


def mean_psnr(test_data, reconstructed_test_data):
    if len(test_data) != len(reconstructed_test_data):
        raise ValueError('Input lists must have the same length')

    psnr_values = []
    for original, reconstructed in zip(test_data, reconstructed_test_data):
        psnr_values.append(peak_signal_noise_ratio(original, reconstructed, data_range=255))
    return np.mean(psnr_values)


def mean_ssim(test_data, reconstructed_test_data):
    if len(test_data) != len(reconstructed_test_data):
        raise ValueError('Input lists must have the same length')

    ssim_values = []
    for original, reconstructed in zip(test_data, reconstructed_test_data):
        ssim_values.append(ssim(original, reconstructed, data_range=255.0))
    return np.mean(ssim_values)
