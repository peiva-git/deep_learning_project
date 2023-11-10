import os
import cv2


def save_images_to_directory(images, directory_path, image_prefix='image', image_format='png'):
    """
    Save images into a specified directory
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    for i, image in enumerate(images):
        image_filename = f"{image_prefix}_{i+1}.{image_format}"
        image_path = os.path.join(directory_path, image_filename)
        cv2.imwrite(image_path, image)

    print(f"Images saved to {directory_path}")


def read_images_from_directory(directory_path):
    """
    Read images from a directory
    """
    images = []

    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    for filename in os.listdir(directory_path):
        image_path = os.path.join(directory_path, filename)

        image = cv2.imread(image_path)
        if image is not None:
            images.append(image)

    return images
