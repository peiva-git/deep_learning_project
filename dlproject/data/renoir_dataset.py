import math
import os
import glob
import cv2
import numpy as np
import tensorflow as tf


class RENOIRDatasetSequence(tf.keras.utils.Sequence):
    def __init__(self, dataset_folder, dataset_type, batch_size, target_size=(5328, 3000)):
        if dataset_type not in ['Mi3_Aligned', 'T3i_Aligned']:
            raise ValueError("Invalid dataset type. Choose 'Mi3_Aligned' or 'T3i_Aligned'.")

        self.dataset_folder = dataset_folder
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.target_size = target_size
        self.test_data_paths = self.__get_test_data_paths()

    def __len__(self):
        return math.ceil(len(self.test_data_paths) / self.batch_size)

    def __getitem__(self, index):
        low = index * self.batch_size
        high = min(low + self.batch_size, len(self.test_data_paths))
        batch_paths = self.test_data_paths[low:high]

        test_data_clean = []
        test_data_noisy = []

        for image_path in batch_paths:
            image_clean = cv2.imread(image_path[0])
            image_noisy = cv2.imread(image_path[1])

            if image_clean is not None and image_noisy is not None:
                image_clean = cv2.resize(image_clean, self.target_size)
                image_noisy = cv2.resize(image_noisy, self.target_size)

                test_data_clean.append(image_clean)
                test_data_noisy.append(image_noisy)

        return np.array(test_data_clean), np.array(test_data_noisy)

    def __get_test_data_paths(self):
        test_data_paths = []

        test_data_folder = os.path.join(self.dataset_folder, self.dataset_type)

        for batch_folder in glob.glob(os.path.join(test_data_folder, 'Batch_*')):
            clean_image_path = glob.glob(os.path.join(batch_folder, '*Reference.bmp'))
            noisy_image_path = glob.glob(os.path.join(batch_folder, '*Noisy.bmp'))

            if clean_image_path and noisy_image_path:
                test_data_paths.append((clean_image_path[0], noisy_image_path[0]))

        return test_data_paths