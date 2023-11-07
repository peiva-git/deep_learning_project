import random
import matplotlib.pyplot as plt


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
