import tensorflow as tf
import numpy as np


class MNISTDatasetBuilder:
    __train_data: np.ndarray
    __test_data: np.ndarray
    __noisy_train_data: np.ndarray
    __noisy_test_data: np.ndarray

    def __init__(self):
        (self.__train_data, ), (self.__test_data) = tf.keras.datasets.mnist.load_data()

    def preprocess_dataset(self):
        self.__train_data = self.__preprocess_array(self.__train_data)
        self.__test_data = self.__preprocess_array(self.__test_data)
        self.__noisy_train_data = self.__add_noise(self.__train_data)
        self.__noisy_test_data = self.__add_noise(self.__test_data)

    @staticmethod
    def __preprocess_array(array: np.ndarray) -> np.ndarray:
        array = array.astype(np.float32) / 255.0
        return np.reshape(array, (len(array), 28, 28, 1))

    @staticmethod
    def __add_noise(array: np.ndarray, noise_factor: float = 0.4) -> np.ndarray:
        noisy_array = array + noise_factor * np.random.normal(
            loc=0.0, scale=1.0, size=array.shape
        )
        return np.clip(noisy_array, 0.0, 1.0)

    @property
    def train_data(self):
        return self.__train_data

    @property
    def test_data(self):
        return self.__test_data

    @property
    def noisy_train_data(self):
        return self.__noisy_train_data

    @property
    def noisy_test_data(self):
        return self.__noisy_test_data
