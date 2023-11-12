import argparse
import os
import glob
import pathlib

import cv2


def extract_and_save_patches(dataset_folder, dataset_type, patch_size=(256, 256), save_patches_folder='./patches'):
    if dataset_type not in ['Mi3_Aligned', 'T3i_Aligned']:
        raise ValueError("Invalid dataset type. Choose 'Mi3_Aligned' or 'T3i_Aligned'.")

    test_data_paths = []

    test_data_folder = os.path.join(dataset_folder, dataset_type)

    for batch_folder in glob.glob(os.path.join(test_data_folder, 'Batch_*')):
        clean_image_path = glob.glob(os.path.join(batch_folder, '*Reference.bmp'))
        noisy_image_path = glob.glob(os.path.join(batch_folder, '*Noisy.bmp'))

        if clean_image_path and noisy_image_path:
            test_data_paths.append((clean_image_path[0], noisy_image_path[0]))

    patches_folder = save_patches_folder

    if not os.path.exists(patches_folder):
        os.makedirs(patches_folder)

    patch_count = 0

    for clean_path, noisy_path in test_data_paths:
        clean_image = cv2.imread(clean_path)
        noisy_image = cv2.imread(noisy_path)

        if clean_image is not None and noisy_image is not None:
            patches_clean = split_into_patches(clean_image, patch_size)
            patches_noisy = split_into_patches(noisy_image, patch_size)

            for clean_patch, noisy_patch in zip(patches_clean, patches_noisy):
                clean_patch_filename = os.path.join(patches_folder, f'clean_patch_{patch_count}.png')
                noisy_patch_filename = os.path.join(patches_folder, f'noisy_patch_{patch_count}.png')

                cv2.imwrite(clean_patch_filename, clean_patch)
                cv2.imwrite(noisy_patch_filename, noisy_patch)

                patch_count += 1


def split_into_patches(image, patch_size):
    patches = []

    height, width, _ = image.shape

    for y in range(0, height, patch_size[0]):
        for x in range(0, width, patch_size[1]):
            patch = image[y:y + patch_size[0], x:x + patch_size[1]]
            if patch.shape[:2] == patch_size:
                patches.append(patch)

    return patches


def preprocess_renoir_dataset(dataset_folder, dataset_type, patch_size=(1024, 1024), save_patches_folder='./patches'):
    # Extract and save patches from the RENOIR dataset
    extract_and_save_patches(dataset_folder, dataset_type, patch_size, save_patches_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_dir',
        help='Root directory of the Renoir dataset',
        type=str,
        required=True
    )
    parser.add_argument(
        '--dataset_type',
        help='Which version of the dataset to pre-process',
        type=str,
        choices=['Mi3_Aligned', 'T3i_Aligned'],
        required=True
    )
    parser.add_argument(
        '--patch_height',
        help='Height of the resulting patches',
        type=int,
        required=True,
        default=1024
    )
    parser.add_argument(
        '--patch_width',
        help='Width of the resulting patches',
        type=int,
        required=True,
        default=1024
    )
    parser.add_argument(
        '--save_dir',
        help='Where to save the resulting patches',
        type=str,
        required=True
    )
    args = parser.parse_args()

    dataset_dir = pathlib.Path(args.dataset_dir)
    save_dir = pathlib.Path(args.save_dir)
    extract_and_save_patches(str(dataset_dir), args.dataset_type, (args.patch_height, args.patch_width), str(save_dir))
