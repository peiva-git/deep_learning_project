from skimage.color import rgb2gray
import tensorflow as tf
import numpy as np


class CIFAR10DatasetBuilder:
    __noisy_train_data: np.ndarray
    __noisy_test_data: np.ndarray

    def __init__(self):
        (self.__train_x, self.__train_y), (self.__test_x, self.__test_y) = tf.keras.datasets.cifar10.load_data()

    def preprocess_dataset_simple_ae(self, noise_factor: float = 0.4):
        self.__train_x = self.__preprocess_array(self.__train_x)
        self.__test_x = self.__preprocess_array(self.__test_x)
        self.__noisy_train_data = self.__add_noise(self.__train_x, noise_factor)
        self.__noisy_test_data = self.__add_noise(self.__test_x, noise_factor)

    def preprocess_dataset_simple_vae(self, noise_factor: float = 0.4):
        self.__train_x = np.array([rgb2gray(train_image) for train_image in self.__train_x], dtype=np.float32)
        self.__test_x = np.array([rgb2gray(test_image) for test_image in self.__test_x], dtype=np.float32)
        self.__train_x = self.__preprocess_array(self.__train_x)
        self.__test_x = self.__preprocess_array(self.__test_x)
        self.__train_x = tf.reshape(self.__train_x, (len(self.__train_x), np.prod(self.__train_x.shape[1:])))
        self.__test_x = tf.reshape(self.__test_x, (len(self.__test_x), np.prod(self.__test_x.shape[1:])))

    @staticmethod
    def __preprocess_array(array: np.ndarray) -> np.ndarray:
        array = array.astype(np.float32) / 255.
        return array

    @staticmethod
    def __add_noise(array: np.ndarray, noise_factor) -> np.ndarray:
        noisy_array = array + noise_factor * np.random.normal(
            loc=0.0, scale=1.0, size=array.shape
        )
        return np.clip(noisy_array, 0.0, 1.0)

    @property
    def train_x(self):
        return self.__train_x

    @property
    def test_x(self):
        return self.__test_x

    @property
    def train_y(self):
        return self.__train_y

    @property
    def test_y(self):
        return self.__test_y

    @property
    def noisy_train_data(self):
        return self.__noisy_train_data

    @property
    def noisy_test_data(self):
        return self.__noisy_test_data
