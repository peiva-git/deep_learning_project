import random

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


def display_random_images(test_data, noisy_test_data, trained_model, num_images=10):
    # Randomly select 10 indices from the test dataset
    random_indices = random.sample(range(len(test_data)), num_images)
    display_images(test_data, noisy_test_data, trained_model, random_indices, num_images)


def display_images(test_data, noisy_test_data, trained_model, image_indexes, num_images):
    # Randomly select 10 indices from the test dataset
    plt.figure(figsize=(15, 4))

    for i, idx in enumerate(image_indexes):
        # Original clean image
        plt.subplot(3, num_images, i + 1)
        plt.imshow(test_data[idx])
        plt.axis('off')

        # Noisy image
        plt.subplot(3, num_images, num_images + i + 1)
        plt.imshow(noisy_test_data[idx])
        plt.axis('off')

        # Predicted output from the autoencoder
        predicted_output = trained_model.model.predict(noisy_test_data[idx].reshape(1, 32, 32, 3))
        plt.subplot(3, num_images, 2 * num_images + i + 1)
        plt.imshow(predicted_output[0])
        plt.axis('off')

    plt.show()
