import os
import glob
import cv2
import numpy as np


class RENOIRDatasetBuilder:
    __test_data_clean: np.ndarray
    __test_data_noisy: np.ndarray

    def __init__(self, dataset_folder, dataset_type):
        if dataset_type not in ['Mi3_Aligned', 'T3i_Aligned']:
            raise ValueError("Invalid dataset type. Choose 'Mi3_Aligned' or 'T3i_Aligned'.")

        self.dataset_folder = dataset_folder
        self.dataset_type = dataset_type

        test_data_folder = os.path.join(dataset_folder, dataset_type)

        self.__test_data_clean = self.__load_test_data(test_data_folder, 'Reference')
        self.__test_data_noisy = self.__load_test_data(test_data_folder, 'Noisy')

    def __load_test_data(self, folder, data_type, target_size=(5328, 3000)):
        test_data = []

        # Store batch names to track the processed batches
        processed_batches = set()

        test_data_paths = glob.glob(os.path.join(folder, 'Batch_*', f'*{data_type}.bmp'))

        for image_path in test_data_paths:
            # Extract the batch name from the image path
            batch_name = os.path.dirname(image_path)

            if batch_name not in processed_batches:
                # Only process one image from each batch
                processed_batches.add(batch_name)

                image = cv2.imread(image_path)

                if image is not None:
                    # Resize the image to the target size
                    image = cv2.resize(image, target_size)
                    # You can add additional preprocessing if needed
                    test_data.append(image)

        return np.array(test_data)

    @property
    def test_data_clean(self):
        return self.__test_data_clean

    @property
    def test_data_noisy(self):
        return self.__test_data_noisy
