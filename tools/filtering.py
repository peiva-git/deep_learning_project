import argparse
import pathlib

import cv2

from dlproject.data.image_utils import read_images_from_directory, save_images_to_directory


def apply_filter(images, filter_type, filter_params=None):
    """
    Apply a specified filter to a set of input images
    """

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir',
        help='Directory from which to read the input images',
        type=str,
        required=True
    )
    parser.add_argument(
        '--output_dir',
        help='Where to write the filtered images',
        type=str,
        required=True
    )
    parser.add_argument(
        '--filter_type',
        help='Which filter to use',
        type=str,
        choices=['Gaussian', 'Bilateral', 'Mean'],
        required=True
    )
    parser.add_argument(
        '--ksize',
        help='ksize parameter for the Gaussian or Mean filters',
        type=int,
        required=False,
        default=5
    )
    parser.add_argument(
        '--sigma',
        help='sigma parameter for the Gaussian filter',
        type=int,
        required=False,
        default=1
    )
    parser.add_argument(
        '--diameter',
        help='diameter parameter for the Bilateral filter',
        type=int,
        required=False,
        default=9
    )
    parser.add_argument(
        '--sigma_color',
        help='sigma color parameter for the Bilateral filter',
        type=int,
        required=False,
        default=75
    )
    parser.add_argument(
        '--sigma_space',
        help='sigma space parameter for the Bilateral filter',
        type=int,
        required=False,
        default=75
    )
    args = parser.parse_args()

    images_dir = pathlib.Path(args.image_dir)
    output_dir = pathlib.Path(args.output_dir)

    loaded_images = read_images_from_directory(str(images_dir))
    if args.filter_type == 'Gaussian':
        result_images = apply_filter(
            loaded_images, 'Gaussian',
            filter_params={'ksize': args.ksize, 'sigma': args.sigma}
        )
    elif args.filter_type == 'Mean':
        result_images = apply_filter(
            loaded_images, 'Mean',
            filter_params={'ksize': args.ksize}
        )
    else:
        result_images = apply_filter(
            loaded_images, 'Bilateral',
            filter_params={'diameter': args.diameter, 'sigma_color': args.sigma_color, 'sigma_space': args.sigma_space}
        )

    save_images_to_directory(result_images, str(output_dir))
