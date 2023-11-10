import cv2
from dlproject.images_file_manager import save_images_to_directory, read_images_from_directory


# Apply a specified filter to a set of input loaded_images
def apply_filter(images, filter_type, filter_params=None):

    filtered_images = []

    for image in images:
        if filter_type == "Gaussian":
            filtered_image = cv2.GaussianBlur(image, (filter_params['ksize'], filter_params['ksize']),
                                              filter_params['sigma'])

        elif filter_type == "Mean":
            filtered_image = cv2.blur(image, (filter_params['ksize'], filter_params['ksize']))

        elif filter_type == "Bilateral":
            filtered_image = cv2.bilateralFilter(image, filter_params['diameter'],
                                                 filter_params['sigma_color'], filter_params['sigma_space'])
        else:
            raise ValueError("Invalid filter type. Supported types: 'Gaussian', 'Mean', 'Bilateral'.")

        filtered_images.append(filtered_image)

    return filtered_images


# Tunable filter parameters
gaussian_params = {'ksize': 5, 'sigma': 1}
mean_params = {'ksize': 5}
bilateral_params = {'diameter': 9, 'sigma_color': 75, 'sigma_space': 75}

# Load loaded_images
loaded_images = read_images_from_directory('C:\\Users\\feder\\Desktop\\DL_project\\noisy_patches')

# Gaussian Filter
gaussian_filtered_images = apply_filter(loaded_images, "Gaussian", filter_params=gaussian_params)

# Mean Filter
mean_filtered_images = apply_filter(loaded_images, "Mean", filter_params=mean_params)

# Bilateral Filter
bilateral_filtered_images = apply_filter(loaded_images, "Bilateral", filter_params=bilateral_params)

# Save filtered loaded_images
save_images_to_directory(gaussian_filtered_images,
                         'C:\\Users\\feder\\Desktop\\DL_project\\traditional_filters_results\\gaussian_filter')
save_images_to_directory(mean_filtered_images,
                         'C:\\Users\\feder\\Desktop\\DL_project\\traditional_filters_results\\mean_filter')
save_images_to_directory(bilateral_filtered_images,
                         'C:\\Users\\feder\\Desktop\\DL_project\\traditional_filters_results\\bilateral_filter')
