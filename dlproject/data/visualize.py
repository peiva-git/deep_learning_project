import random

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def show_encoder_scatter_plot(x_test: np.ndarray, y_test: np.ndarray, encoder: tf.keras.Model):
    x_test_encoded = encoder.predict(x_test)[2]
    plt.figure(figsize=(6, 6))
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
    plt.colorbar()
    plt.show()


def show_latent_plane_sampled_points(decoder: tf.keras.Model,
                                     x_sampling_interval: (int, int),
                                     y_sampling_interval: (int, int),
                                     number_of_figures: int = 15,
                                     figure_size: int = 28):
    figure = np.zeros((figure_size * number_of_figures, figure_size * number_of_figures))
    grid_x = np.linspace(x_sampling_interval[0], x_sampling_interval[1], number_of_figures)
    grid_y = np.linspace(y_sampling_interval[0], y_sampling_interval[1], number_of_figures)
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder(z_sample)
            digit = tf.reshape(x_decoded[0], (figure_size, figure_size))
            figure[
            i * figure_size: (i + 1) * figure_size,
            j * figure_size: (j + 1) * figure_size
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
