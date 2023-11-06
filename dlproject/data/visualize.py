import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np


def show_mnist_scatter_plot(x_test: np.ndarray, y_test: np.ndarray, encoder: tf.keras.Model, batch_size: int):
    x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
    plt.figure(figsize=(6, 6))
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
    plt.colorbar()
    plt.show()


def show_latent_plane_sampling_digits(decoder: tf.keras.Model, number_of_digits: int = 15, digit_size: int = 28):
    figure = np.zeros((digit_size * number_of_digits, digit_size * number_of_digits))
    grid_x = np.linspace(-15, 15, number_of_digits)
    grid_y = np.linspace(-15, 15, number_of_digits)
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([xi, yi])
            x_decoded = decoder(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size
            ] = digit
    plt.figure(figsize=(10, 10))
    plt.imshow(figure)
    plt.show()
